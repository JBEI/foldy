import pandas as pd
import json

import numpy as np
from flask import request
from flask_restx import Resource, fields
from flask_jwt_extended import jwt_required
from flask_restx import Namespace
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import BadRequest
from sklearn.ensemble import RandomForestRegressor

from app.models import Fold
from app.util import FoldStorageManager
from app.helpers.mutation_util import (
    maybe_get_seq_id_error_message,
    process_and_validate_evolve_input_files,
    get_train_and_test_mutant_seq_ids,
)

ns = Namespace("evolve_views", decorators=[jwt_required(fresh=True)])

upload_parser = ns.parser()
upload_parser.add_argument(
    "activity_file", type=FileStorage, location="files", required=True
)
upload_parser.add_argument("fold_id", type=str, location="form", required=True)
upload_parser.add_argument("embedding_paths", type=str, location="form", required=True)


@ns.route("/evolve")
class EvolveResource(Resource):
    @ns.expect(upload_parser)
    # @ns.consumes('multipart/form-data')
    def post(self):
        args = upload_parser.parse_args()

        # Get form data
        fold_id = int(args["fold_id"])
        embedding_paths = json.loads(args["embedding_paths"])
        activity_file = args["activity_file"]

        fold = Fold.get_by_id(fold_id)
        if not fold:
            raise BadRequest(f"Fold {fold_id} not found")
        wt_aa_seq = fold.sequence

        # Get activity file from request
        if "activity_file" not in request.files:
            return {"error": "No activity file provided"}, 400
        activity_file = request.files["activity_file"]

        # Read activity data from uploaded Excel file
        raw_activity_df = pd.read_excel(activity_file)
        # Initialize storage manager
        fsm = FoldStorageManager()
        fsm.setup()

        # Read and merge all embedding CSVs
        embedding_dfs = []
        chunk_size = 10000  # Adjust based on memory constraints

        for path in embedding_paths:
            # Get the CSV content as a string
            csv_blob = fsm.storage_manager.get_blob(fold_id, path)

            with csv_blob.open("r") as csv_f:
                # Create chunks iterator
                chunks = pd.read_csv(csv_f, chunksize=chunk_size)

                # Process each chunk
                path_dfs = []
                for chunk in chunks:
                    path_dfs.append(chunk)

                # Combine chunks for this path
                if path_dfs:
                    embedding_dfs.append(pd.concat(path_dfs, ignore_index=True))

        # Combine all embeddings
        raw_embedding_df = pd.concat(embedding_dfs, ignore_index=True)

        activity_df, embedding_df = process_and_validate_evolve_input_files(
            raw_activity_df, raw_embedding_df
        )

        train_mutants, test_mutants = get_train_and_test_mutant_seq_ids(
            activity_df, embedding_df
        )

        # Convert JSON strings to lists and stack them into a 2D numpy array
        X_train = np.vstack(
            [json.loads(x) for x in embedding_df.loc[activity_df.index].embedding]
        )
        y_train = activity_df.activity.to_numpy()

        # print(X_train, flush=True)
        # print(type(X_train), flush=True)
        # print(type(X_train.iloc[0]), flush=True)

        model = RandomForestRegressor(
            n_estimators=100,
            criterion="friedman_mse",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=1.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=1,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
        )
        model.fit(X_train, y_train)

        # # make predictions on train data
        # y_pred_train = model.predict(X_train)
        # y_std_train = np.zeros(len(y_pred_train))

        # # make predictions on test data
        # y_pred_test = model.predict(X_test)

        return {"train_mutants": train_mutants, "test_mutants": test_mutants}
