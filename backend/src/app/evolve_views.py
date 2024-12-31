import pandas as pd

from flask import request
from flask_restx import Resource, fields
from flask_jwt_extended import jwt_required
from flask_restx import Namespace

from app.util import FoldStorageManager


ns = Namespace("evolve_views", decorators=[jwt_required(fresh=True)])

evolve_input = ns.model('EvolveInput', {
    'fold_id': fields.Integer(required=True, description='ID of the fold'),
    'embedding_paths': fields.List(fields.String, required=True, description='Paths to embedding CSV files'),
})

@ns.route('/evolve')
class EvolveResource(Resource):
    @ns.expect(evolve_input)
    def post(self):
        # Get JSON payload
        data = request.get_json()
        fold_id = data['fold_id']
        embedding_paths = data['embedding_paths']
        
        # Get activity file from request
        if 'activity_file' not in request.files:
            return {'error': 'No activity file provided'}, 400
        activity_file = request.files['activity_file']
        
        # Read activity data from uploaded Excel file
        activity_df = pd.read_excel(activity_file)
        
        # Initialize storage manager
        fsm = FoldStorageManager()
        fsm.setup()
        
        # Read and merge all embedding CSVs
        embedding_dfs = []
        chunk_size = 10000  # Adjust based on memory constraints
        
        for path in embedding_paths:
            # Get the CSV content as a string
            csv_blob = fsm.storage_manager.get_blob(path)
            
            with csv_blob.open('r') as csv_f:
                # Create chunks iterator
                chunks = pd.read_csv(
                    csv_f,
                    chunksize=chunk_size
                )
                
                # Process each chunk
                path_dfs = []
                for chunk in chunks:
                    path_dfs.append(chunk)
                
                # Combine chunks for this path
                if path_dfs:
                    embedding_dfs.append(pd.concat(path_dfs, ignore_index=True))
        
        # Combine all embeddings
        embedding_df = pd.concat(embedding_dfs, ignore_index=True)
        
        # Get mutant sets
        activity_mutants = set(activity_df['mutant'])
        embedding_mutants = set(embedding_df['seq_id'])
        
        # Calculate overlap and test sets
        train_mutants = list(activity_mutants.intersection(embedding_mutants))
        test_mutants = list(embedding_mutants - activity_mutants)
        
        return {
            'train_mutants': train_mutants,
            'test_mutants': test_mutants
        }


