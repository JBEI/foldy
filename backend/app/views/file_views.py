import io
import logging
import re
from typing import (
    Any,
    BinaryIO,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

from flask import (
    Response,
    current_app,
    make_response,
    request,
    send_file,
    stream_with_context,
)
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource, fields
from sqlalchemy.sql.elements import and_
from werkzeug.exceptions import BadRequest

from app.api_fields import fold_file_zip_fields
from app.extensions import db
from app.helpers.fold_storage_manager import FoldStorageManager
from app.models import Dock, Fold, Invokation

ns = Namespace("file_views", decorators=[jwt_required(fresh=True)])


# fold_file_zip_fields imported from api_fields.py


# @compress.compressed()
@ns.route("/fold_file_zip")
class FoldFileZipResource(Resource):
    @ns.expect(fold_file_zip_fields)
    def post(self):
        """Get zip file containing multiple fold files.

        Returns:
            Zip file with requested fold files
        """
        manager = FoldStorageManager()
        manager.setup()

        json_data = request.get_json()
        fold_ids = json_data["fold_ids"]
        relative_fpath = json_data["relative_fpath"]
        output_dirname = json_data["output_dirname"]
        flatten_filepath = bool(json_data.get("flatten_filepath", False))
        use_fold_name = bool(json_data.get("use_fold_name", False))

        logging.info(
            f"API received flatten_filepath={flatten_filepath} (type: {type(flatten_filepath)}) use_fold_name={use_fold_name} (type: {type(use_fold_name)})"
        )

        return send_file(
            manager.get_fold_file_zip(
                fold_ids,
                relative_fpath,
                output_dirname,
                flatten_filepath,
                use_fold_name,
            ),
            mimetype="application/octet-stream",
            download_name="fold_files.zip",
            as_attachment=True,
        )


@ns.route("/dock_sdf/<int:fold_id>/<string:ligand_name>")
class DockSdfResource(Resource):
    def post(self, fold_id: int, ligand_name: str):
        """Get SDF file for a specific dock.

        Args:
            fold_id: ID of the fold/protein
            ligand_name: Name of the ligand

        Returns:
            File response with SDF data

        Raises:
            BadRequest: If dock is not found
        """
        print(f"Finding dock for {fold_id} for {ligand_name}", flush=True)
        dock = (
            db.session.query(Dock)
            .filter(
                and_(
                    Dock.ligand_name == ligand_name,
                    Dock.receptor_fold_id == fold_id,
                )
            )
            .first()
        )

        if not dock:
            raise BadRequest(f"Dock for fold id {fold_id} ligand name {ligand_name} not found.")

        manager = FoldStorageManager()
        manager.setup()
        sdf_str = manager.get_dock_sdf(dock)
        return send_file(
            io.BytesIO(sdf_str),
            mimetype="application/octet-stream",
            download_name=f"{ligand_name}.sdf",
            as_attachment=True,
        )


@ns.route("/file/list/<int:fold_id>")
class FoldFileResource(Resource):
    def get(self, fold_id: int) -> List[Dict[str, Any]]:
        """List all files for a given fold.

        Args:
            fold_id: ID of the fold

        Returns:
            List of dictionaries with file information
        """
        manager = FoldStorageManager()
        manager.setup()
        return manager.storage_manager.list_files(fold_id)


@ns.route("/file/download/<int:fold_id>/<path:subpath>")
class FileDownloadResource(Resource):
    def get(self, fold_id: int, subpath: str) -> Union[Response, Tuple[Dict[str, str], int]]:
        """Download a file from fold storage with streaming.

        Args:
            fold_id: ID of the fold
            subpath: Path to file within fold storage

        Returns:
            Streaming response with file data or error message with status code
        """
        print(f"Starting download for {subpath}...")
        manager = FoldStorageManager()
        manager.setup()

        try:
            blob = manager.storage_manager.get_blob(fold_id, subpath)
        except BadRequest as e:
            return {"message": str(e)}, 400
        except Exception as e:
            print(f"Error accessing blob: {str(e)}")
            return {"message": "Error accessing file"}, 500

        fname = subpath.split("/")[-1]

        def generate() -> Generator[bytes, None, None]:
            """Generate file chunks for streaming.

            Yields:
                Chunks of file data
            """
            try:
                with blob.open("rb") as f:
                    while True:
                        chunk = f.read(1024 * 1024)  # 1MB chunks
                        if not chunk:
                            break
                        yield cast(bytes, chunk)  # blob opened in 'rb' mode guarantees bytes
            except Exception as e:
                print(f"Error during file streaming: {str(e)}")
                return

        headers = {
            "Content-Disposition": f"attachment; filename={fname}",
            "Content-Type": "application/octet-stream",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "Keep-Alive": "timeout=1800",  # 30 minutes
            "Pragma": "no-cache",
            "Expires": "0",
        }

        # Only add Content-Length if blob.size is available
        if hasattr(blob, "size") and blob.size is not None:
            headers["Content-Length"] = str(blob.size)

        try:
            return Response(
                stream_with_context(generate()),
                headers=headers,
                direct_passthrough=True,
                mimetype="application/octet-stream",
            )
        except Exception as e:
            print(f"Error creating response: {str(e)}")
            return {"message": "Error streaming file"}, 500


@ns.route("/fancyfile/download/<int:fold_id>/<path:subpath>")
class FancyFileDownloadResource(Resource):
    def get(self, fold_id: int, subpath: str) -> Union[Response, Tuple[Dict[str, str], int]]:
        """Download a file with proper range header support for browser native downloads.

        Args:
            fold_id: ID of the fold
            subpath: Path to file within fold storage

        Query Parameters:
            filename: Optional custom filename for download (overrides original filename)

        Returns:
            Response with proper range headers for browser downloads or error message with status code
        """
        print(f"Starting fancy download for {subpath}...")
        manager = FoldStorageManager()
        manager.setup()

        try:
            blob = manager.storage_manager.get_blob(fold_id, subpath)
        except BadRequest as e:
            return {"message": str(e)}, 400
        except Exception as e:
            print(f"Error accessing blob: {str(e)}")
            return {"message": "Error accessing file"}, 500

        # Check if a custom filename is provided via query parameter
        custom_filename = request.args.get("filename")
        fname = custom_filename if custom_filename else subpath.split("/")[-1]

        # Get file size
        file_size: int = 0
        if hasattr(blob, "size") and blob.size is not None:
            # Ensure file_size is an int, not a callable
            size_value = blob.size
            if callable(size_value):
                result = size_value()
                if not isinstance(result, int):
                    raise TypeError(f"Expected int from size(), got {type(result)}")
                file_size = result
            else:
                file_size = int(size_value)
        else:
            # If size is not available, try to get it by opening the blob
            try:
                with blob.open("rb") as f:
                    f.seek(0, 2)  # Seek to end
                    file_size = f.tell()
            except Exception as e:
                print(f"Could not determine file size: {str(e)}")
                return {"message": "Could not determine file size"}, 500

        assert isinstance(file_size, int), "file_size must be an integer"

        # Parse Range header
        range_header = request.headers.get("Range")
        start = 0
        end = file_size - 1

        if range_header:
            # Parse range header like "bytes=0-1023" or "bytes=1024-"
            range_match = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if range_match:
                start = int(range_match.group(1))
                if range_match.group(2):
                    end = int(range_match.group(2))
                else:
                    end = file_size - 1

                # Ensure range is valid
                if start >= file_size:
                    return make_response("Range Not Satisfiable", 416)
                if end >= file_size:
                    end = file_size - 1

        content_length = end - start + 1

        def generate_range() -> Generator[bytes, None, None]:
            """Generate file chunks for range requests.

            Yields:
                Chunks of file data within the requested range
            """
            try:
                with blob.open("rb") as f:
                    f.seek(start)
                    bytes_remaining = content_length
                    while bytes_remaining > 0:
                        chunk_size = min(
                            1024 * 1024, bytes_remaining
                        )  # 1MB chunks or remaining bytes
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        bytes_remaining -= len(chunk)
                        yield cast(bytes, chunk)
            except Exception as e:
                print(f"Error during range streaming: {str(e)}")
                return

        # Build headers for range response
        headers = {
            "Content-Disposition": f"attachment; filename={fname}",
            "Content-Type": "application/octet-stream",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }

        # Add range headers if this is a range request
        if range_header:
            headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
            status_code = 206  # Partial Content
        else:
            status_code = 200

        try:
            response = Response(
                stream_with_context(generate_range()),
                status=status_code,
                headers=headers,
                direct_passthrough=True,
                mimetype="application/octet-stream",
            )
            return response
        except Exception as e:
            print(f"Error creating fancy response: {str(e)}")
            return {"message": "Error streaming file"}, 500
