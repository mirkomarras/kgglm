import numpy as np
from tqdm import tqdm

from kgglm.knowledge_graphs.kg_cafe import CAFEKnowledgeGraph
from kgglm.knowledge_graphs.kg_macros import (
    ENTITY_LIST,
    INTERACTION,
    PRODUCT,
    RELATION_LIST,
    USER,
)
from kgglm.model.rl.CAFE.cafe_utils import (
    load_kg,
    load_labels,
    save_embed,
    save_kg,
    save_path_count,
    save_user_products,
)
from kgglm.model.rl.CAFE.parser import parse_args
from kgglm.model.rl.PGPR.pgpr_utils import TRANSE, load_embed


def load_kg_embedding(dataset: str):
    """Note that entity embedding is of size [vocab_size+1, d]."""
    print(">>> Load KG embeddings ...")
    state_dict = load_embed(dataset, TRANSE)
    embeds = dict()
    # Load entity embeddings
    for entity in ENTITY_LIST[dataset]:
        embeds[entity] = (
            state_dict[entity].cpu().data.numpy()[:-1]
        )  # remove last dummy embed with 0 values.
        print(f">>> {entity}: {embeds[entity].shape}")
    for rel in RELATION_LIST[dataset]:
        # if dataset=='lfm1m' and rel=='watched':
        #     rel='listened'
        embeds[rel] = state_dict[rel].cpu().data.numpy()[0]
    return embeds


def compute_top100_items(dataset):
    embeds = load_embed(dataset, TRANSE)
    user_embed = embeds[USER].cpu()
    product_embed = embeds[PRODUCT].cpu()
    purchase_embed = embeds[INTERACTION[dataset]].cpu()
    scores = np.dot(user_embed + purchase_embed, product_embed.T)
    user_products = np.argsort(scores, axis=1)  # From worst to best
    best100 = user_products[:, -100:][:, ::-1]
    print(best100.shape)
    return best100


def estimate_path_count(args):
    kg = load_kg(args.dataset)
    num_mp = len(kg.metapaths)
    train_labels = load_labels(args.dataset, "train")
    counts = {}
    pbar = tqdm(total=len(train_labels))
    for uid in train_labels:
        counts[uid] = np.zeros(num_mp)
        for pid in train_labels[uid]:
            for mpid in range(num_mp):
                cnt = kg.count_paths_with_target(mpid, uid, pid, 50)
                counts[uid][mpid] += cnt
        counts[uid] = counts[uid] / len(train_labels[uid])
        pbar.update(1)
    save_path_count(args.dataset, counts)


def main(args):
    # Run following code to extract embeddings from state dict.
    # ========== BEGIN ========== #
    embeds = load_kg_embedding(args.dataset)
    save_embed(args.dataset, embeds)
    # =========== END =========== #
    # Run following codes to generate KnowledgeGraph object.
    # ========== BEGIN ========== #
    kg = CAFEKnowledgeGraph(args.dataset)
    save_kg(args.dataset, kg)
    # =========== END =========== #

    # Run following codes to generate top100 items for each user.
    # ========== BEGIN ========== #
    best100 = compute_top100_items(args.dataset)
    save_user_products(args.dataset, best100, "pos")
    # =========== END =========== #

    # Run following codes to estimate paths count.
    # ========== BEGIN ========== #
    estimate_path_count(args)
    # =========== END =========== #


if __name__ == "__main__":
    args = parse_args()
    main(args)
