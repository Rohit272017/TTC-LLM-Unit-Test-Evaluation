---
dataset_info:
  features:
  - name: ID
    dtype: string
  - name: Language
    dtype: string
  - name: Repository Name
    dtype: string
  - name: File Name
    dtype: string
  - name: File Path in Repository
    dtype: string
  - name: File Path for Unit Test
    dtype: string
  - name: Code
    dtype: string
  - name: Unit Test - (Ground Truth)
    dtype: string
  - name: Code Url
    dtype: string
  - name: Test Code Url
    dtype: string
  - name: Commit Hash
    dtype: string
  splits:
  - name: train
    num_bytes: 64936654
    num_examples: 2629
  - name: eval
    num_bytes: 4873297
    num_examples: 200
  - name: test
    num_bytes: 14853471
    num_examples: 658
  download_size: 23026371
  dataset_size: 84663422
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: eval
    path: data/eval-*
  - split: test
    path: data/test-*
---
