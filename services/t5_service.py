import logging
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast


class T5Service:
    def __init__(self):
        self.__init_logger()

        self.__model: str = "UrukHan/t5-russian-spell"
        self.__task_prefix = "Spell correct: "
        self.__max_input = 4096

        self.__tokenizer = T5TokenizerFast.from_pretrained(self.__model)
        self.__model = AutoModelForSeq2SeqLM.from_pretrained(self.__model)

        self.__logger.debug(f"T5Service initialized!")

    def normalize(self, texts: list[str]) -> list[str]:
        encoded = self.__tokenizer(
            [self.__task_prefix + text for text in texts],
            padding="longest",
            max_length=self.__max_input,
            truncation=True,
            return_tensors="pt",
        )

        predicts = self.__model.generate(**encoded)
        return self.__tokenizer.batch_decode(predicts, skip_special_tokens=True)

    def __init_logger(self):
        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(logging.DEBUG)
