import io
from flask import Response, stream_with_context, send_file
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource
from flask_restx import fields
from werkzeug.exceptions import BadRequest

from app.util import FoldStorageManager

ns = Namespace("file_views", decorators=[jwt_required(fresh=True)])

fold_pdb_fields = ns.model(
    "FoldPdb",
    {
        "pdb_string": fields.String(readonly=True),
    },
)


# @compress.compressed()
@ns.route("/fold_pdb/<int:fold_id>/<int:model_number>")
class FoldResource(Resource):
    @ns.marshal_with(fold_pdb_fields)
    def get(self, fold_id, model_number):
        manager = FoldStorageManager()
        manager.setup()
        return {"pdb_string": manager.get_fold_pdb(fold_id, model_number)}


fold_file_zip_fields = ns.model(
    "FoldPdbZip",
    {
        "fold_ids": fields.List(fields.Integer(required=True)),
        "relative_fpath": fields.String(),
        "output_dirname": fields.String(),
    },
)


# @compress.compressed()
@ns.route("/fold_file_zip")
class FoldPdbZipResource(Resource):
    @ns.expect(fold_file_zip_fields)
    def post(self):
        manager = FoldStorageManager()
        manager.setup()

        return send_file(
            manager.get_fold_file_zip(
                request.get_json()["fold_ids"],
                request.get_json()["relative_fpath"],
                request.get_json()["output_dirname"],
            ),
            mimetype="application/octet-stream",
            download_name="fold_pdbs.zip",
            as_attachment=True,
        )


@ns.route("/fold_pkl/<int:fold_id>/<int:model_number>")
class FoldPklResource(Resource):
    def post(self, fold_id, model_number):
        manager = FoldStorageManager()
        manager.setup()
        pkl_byte_str = manager.get_fold_pkl(fold_id, model_number)
        # TODO: Stream!
        # https://flask.palletsprojects.com/en/1.1.x/patterns/streaming/
        # https://www.reddit.com/r/Python/comments/dha9kw/flask_send_large_file/
        return send_file(
            io.BytesIO(pkl_byte_str),
            mimetype="application/octet-stream",
            download_name="model.pkl",
            as_attachment=True,
        )


@ns.route("/dock_sdf/<int:fold_id>/<string:ligand_name>")
class FoldPklResource(Resource):
    def post(self, fold_id, ligand_name):
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
            raise BadRequest(
                f"Dock for fold id {fold_id} ligand name {ligand_name} not found."
            )

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
    def get(self, fold_id):
        manager = FoldStorageManager()
        manager.setup()
        return manager.storage_manager.list_files(fold_id)


# @ns.route("/file/download/<int:fold_id>/<path:subpath>")
# class FileDownloadResource(Resource):
#     def post(self, fold_id, subpath):
#         # TODO: test this.
#         print(f"Fetching {subpath}...")
#         manager = FoldStorageManager()
#         manager.setup()
#         sdf_str = manager.storage_manager.get_binary(fold_id, subpath)
#         fname = subpath.split("/")[-1]
#         return send_file(
#             io.BytesIO(sdf_str),
#             mimetype="application/octet-stream",
#             download_name=fname,
#             as_attachment=True,
#         )


@ns.route("/file/download/<int:fold_id>/<path:subpath>")
class FileDownloadResource(Resource):
    def get(self, fold_id, subpath):
        print(f"Fetching {subpath}...")
        manager = FoldStorageManager()
        manager.setup()

        try:
            blob = manager.storage_manager.get_blob(fold_id, subpath)
        except BadRequest as e:
            return {"message": str(e)}, 400

        fname = subpath.split("/")[-1]

        def generate():
            with blob.open("rb") as f:
                while True:
                    chunk = f.read(8192)  # 8KB chunks
                    if not chunk:
                        break
                    yield chunk

        headers = {
            "Content-Disposition": f"attachment; filename={fname}",
            "Content-Type": "application/octet-stream",
        }

        return Response(
            stream_with_context(generate()), headers=headers, direct_passthrough=True
        )
