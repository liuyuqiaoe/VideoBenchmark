# VideoBenchmark

## Repository Structure

```
VideoBenchmark/
├── MRAG/                    # MRAG Benchmark evaluation
├── models/                  # models to construct some encoder (VLM2vec)
├── video_retrieval/         # Video/Image retrieval and indexing system
└── .gitmodules             # Git submodules configuration
```

## Getting Started

### Prerequisites

- Python 3.10
- PyTorch 2.6.0


### Installation

```bash
git clone https://github.com/liuyuqiaoe/VideoBenchmark.git
cd VideoBenchmark
```

### Usage of video_retrival module

#### encoders
Basic usage is specified in the main function of each encoder.

#### stanford40action
Codes for stanford40action benchmark evaluation, including generators for text description generation, result analysis tools and other utils.

#### retrieval system
Core of the project. 
- video_indexer.py -> retrieval system
- qa_tester.py -> evaluation on different dataset. The implementation of retrieval function of our system can be seen here.
- indexing.py -> indexing images and videos. The implementation of indexing function of our system can be seen here.
- patch_analysis.py -> module for patch analysis.

The full use of llave encoder based retrieval and indexing can be implemented with only the video_indexer.py and encoders. You can check qa_tester.py and indexing.py to learn more detail on its usage.
```python
from video_retrieval.encoders.llava_qwen_encoder import LLaVAQwenEncoder
from video_retrieval.video_indexer import LanceDBVideoRetriever
from PIL import Image

llava_encoder = LLaVAQwenEncoder()
# check video_indexer to find more types
similarity_type = "colbert_maxsim_mean" 

retriever = LanceDBVideoRetriever(encoder=llava_encoder, db_path=db_path_s40a_llava, table_name="image_10patches_embeddings")

# using image to retrieve image/video
img2img_results = retriever.search_image(
    images=images, # List[Image.Image], query images
    top_k=topk_lst, # List[int], num of retrieved items to be returned for each query
    return_all=False, # ignore it, always False
    similarity_type=similarity_type,
    return_gt=False # List[gt_image_id], you can ignore it
)

# using text to retrieve image/video
text2img_results = retriever.search(
    queries=queries, # List[str], query texts
    top_k=topk_lst,
    return_all=False,
    similarity_type=similarity_type
)
```


