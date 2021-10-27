# KoBERT-Relation

- 감염병과 관련이 있는 기사인지 분류하는 프로그램
- 🤗Huggingface Tranformers🤗 라이브러리를 이용하여 구현

## Dependencies

- torch==1.9.1
- transformers==4.11.0
- tensorboardX>=2.0

### Training

```bash
$ python3 run.py
```

## Results
```
accuracy 94.18
f1 score 94.15
```

## References

- [KoBERT-Transformers](https://github.com/monologg/KoBERT-Transformers)
- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
