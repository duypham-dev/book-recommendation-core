"""
Training script for Hybrid Implicit ALS + SBERT Recommender

Usage:
    python train.py --evaluate --alpha 0.6
    python train.py --als-factors 64 --als-iterations 30
"""
import argparse
import sys
from pathlib import Path

from src.data.db_loader import DatabaseLoader
from src.models.hybrid_recommender import HybridRecommender
from src.utils.config import get_settings
from src.utils.logging_config import logger
from src.utils.evaluation import RecommenderEvaluator, DataSplitter


def main():
    # ==================== Parse Arguments ====================
    parser = argparse.ArgumentParser(description='Train Hybrid Implicit ALS + SBERT Recommender')
    
    # Data args
    parser.add_argument('--db-uri', type=str, help='Database URI')
    parser.add_argument('--schema', type=str, default='book_recommendation_system')
    
    # Model args
    parser.add_argument('--alpha', type=float, default=0.4,
                       help='ALS weight (0-1), SBERT weight = 1-alpha')
    parser.add_argument('--als-factors', type=int, default=64,
                       help='ALS latent factors')
    parser.add_argument('--als-iterations', type=int, default=30,
                       help='ALS training iterations')
    parser.add_argument('--als-regularization', type=float, default=0.01,
                       help='ALS regularization parameter')
    parser.add_argument('--device', type=str, default=None,
                       help='Device: cuda or cpu (auto-detect if None)')
    
    # Training args
    parser.add_argument('--artifacts-dir', type=str, default='./artifacts',
                       help='Directory to save models')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation on test set')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                       help='Test set ratio for evaluation (default: 0.2)')
    
    args = parser.parse_args()
    
    # ==================== Load Settings ====================
    settings = get_settings()
    db_uri = args.db_uri or settings.db_uri
    db_schema = args.schema
    artifacts_dir = Path(args.artifacts_dir)
    
    logger.info("="*70)
    logger.info("🚀 TRAINING HYBRID IMPLICIT ALS + SBERT RECOMMENDER")
    logger.info("="*70)
    logger.info(f"Database: {db_uri}")
    logger.info(f"Schema: {db_schema}")
    logger.info(f"Alpha (ALS weight): {args.alpha}")
    logger.info(f"ALS Factors: {args.als_factors}")
    logger.info(f"ALS Iterations: {args.als_iterations}")
    logger.info(f"ALS Regularization: {args.als_regularization}")
    logger.info(f"Device: {args.device or 'auto-detect'}")
    logger.info(f"Artifacts: {artifacts_dir}")
    logger.info("="*70)
    
    # ==================== Load Data ====================
    logger.info("\n📊 Loading data from database...")
    loader = DatabaseLoader(db_uri, db_schema)
    books_df, interactions_df = loader.load_all()
    
    logger.info(f"✅ Loaded {len(books_df)} books")
    logger.info(f"✅ Loaded {len(interactions_df)} interactions")
    logger.info(f"   Users: {interactions_df['user_id'].nunique()}")
    logger.info(f"   Books: {interactions_df['book_id'].nunique()}")
    
    # ==================== Train/Test Split (if evaluating) ====================
    if args.evaluate:
        logger.info(f"\n📈 Splitting data (test_ratio={args.test_ratio})...")
        train_df, test_df = DataSplitter.temporal_split(
            interactions_df, 
            test_ratio=args.test_ratio
        )
        logger.info(f"   Train: {len(train_df)} interactions")
        logger.info(f"   Test:  {len(test_df)} interactions")
    else:
        train_df = interactions_df
        test_df = None
    
    # ==================== Train Model ====================
    logger.info("\n🔧 Initializing Hybrid Implicit ALS + SBERT Recommender...")
    recommender = HybridRecommender(
        alpha=args.alpha,
        als_factors=args.als_factors,
        als_iterations=args.als_iterations,
        als_regularization=args.als_regularization,
        device=args.device
    )
    
    logger.info("\n🚂 Training models...")
    recommender.train(books_df, train_df)
    
    # ==================== Evaluate (Optional) ====================
    if args.evaluate and test_df is not None:
        logger.info("\n📊 EVALUATION ON TEST SET")
        logger.info("="*70)
        
        metrics = RecommenderEvaluator.compute_metrics(
            recommender, 
            test_df, 
            train_df,
            k=10
        )
        
        logger.info(f"Hit Rate@10:   {metrics['HR@K']:.4f}")
        logger.info(f"NDCG@10:       {metrics['NDCG@K']:.4f}")
        logger.info(f"Users evaluated: {metrics['users_evaluated']}")
        logger.info(f"Users skipped:   {metrics['users_skipped']}")
        
        # Compute coverage
        coverage = RecommenderEvaluator.compute_coverage(
            recommender,
            train_df,
            num_samples=100
        )
        logger.info(f"Coverage:      {coverage:.4f}")
        logger.info("="*70)
        
        # Sample recommendations
        test_users = test_df['user_id'].unique()[:3]
        logger.info("\n📚 SAMPLE RECOMMENDATIONS")
        logger.info("="*70)
        for user_id in test_users:
            RecommenderEvaluator.print_sample_recommendations(
                recommender, books_df, user_id, limit=5
            )
    
    # ==================== Save Model ====================
    logger.info(f"\n💾 Saving model to {artifacts_dir}...")
    recommender.save(artifacts_dir)
    logger.info("✅ Model saved successfully!")
    
    # ==================== Summary ====================
    logger.info("\n" + "="*70)
    logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*70)
    logger.info(f"Model saved to: {artifacts_dir}")
    logger.info(f"Alpha (ALS:SBERT): {args.alpha}:{1-args.alpha}")
    logger.info("\nTo start the server:")
    logger.info(f"  python server.py")
    logger.info("\nTo test the API:")
    logger.info(f"  python test_api.py")
    logger.info("="*70)


if __name__ == "__main__":
    main()
