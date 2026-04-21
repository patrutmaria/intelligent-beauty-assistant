"""
Train VGAE model.  Run: python train_vgae.py
Saves checkpoint to models/vgae_beauty.pt (~1 minute).
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Train VGAE for beauty recommendations")
    parser.add_argument("--data_dir",   default="data/")
    parser.add_argument("--model_path", default="models/vgae_beauty.pt")
    parser.add_argument("--hidden",     type=int,   default=64)
    parser.add_argument("--latent",     type=int,   default=32)
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--lr",         type=float, default=0.01)
    parser.add_argument("--beta",       type=float, default=1.0)
    parser.add_argument("--plot",       action="store_true")
    args = parser.parse_args()

    try:
        import torch
        import torch_geometric
    except ImportError as e:
        print(f"\n[ERROR] Missing dependency: {e}")
        print("Install with:  pip install torch torch-geometric pandas scikit-learn")
        sys.exit(1)

    from graph.graph_builder import BeautyGraphBuilder
    from graph.trainer import train_vgae

    print(f"\nLoading data from '{args.data_dir}' ...")
    builder = BeautyGraphBuilder(args.data_dir)
    graph_data, node_id_map, node_types = builder.build()

    print(f"Graph built: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    print(f"  Products: {len(builder.get_product_node_ids())}")
    print(f"  Looks:    {len(builder.get_look_node_ids())}")
    print(f"  Users:    {len(builder.get_user_node_ids())}")

    model, embeddings = train_vgae(
        graph_data,
        hidden_channels=args.hidden,
        out_channels=args.latent,
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        model_save_path=args.model_path,
        verbose=True,
    )

    if args.plot:
        _plot_embeddings(embeddings, node_types, args.model_path)

    print("Done. Start the server: python api.py")


def _plot_embeddings(embeddings, node_types, model_path):
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        emb_np = embeddings.numpy()
        pca = PCA(n_components=2)
        coords = pca.fit_transform(emb_np)

        type_colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red"}
        type_labels = {0: "User", 1: "Product", 2: "Look", 3: "Ingredient"}

        plt.figure(figsize=(9, 7))
        for t, color in type_colors.items():
            mask = [i for i, nt in enumerate(node_types) if nt == t]
            if mask:
                plt.scatter(coords[mask, 0], coords[mask, 1],
                            c=color, label=type_labels[t], alpha=0.7, s=40)

        plt.title("VGAE Latent Embeddings – PCA (2D)")
        plt.legend()
        plt.tight_layout()

        plot_path = model_path.replace(".pt", "_embeddings.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Embedding plot saved -> {plot_path}")
    except Exception as e:
        print(f"[WARN] Could not generate plot: {e}")


if __name__ == "__main__":
    main()
