from typing import (
    Optional,
    Iterator,
    Iterable,
    cast,
    TypedDict,
    overload,
    List,
)
from pathlib import Path
import abc
import csv
import datetime
import enum
import math
import random
import weakref

class Sample:
    """기본 샘플 클래스"""
    def __init__(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
        species: Optional[str] = None,
    ) -> None:
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width
        self.species = species
        self.classification = Optional[str] = None

    def __repr__(self) -> str:
        if self.species is None:
            known_unknown = "UnknownSample"
        else:
            known_unknown = "KnownSample"
        if self.classification is None:
            classification = ""
        else:
            classification = f", {self.classification}"

    def classifiy(self, classification: str) -> None:
        self.classification = classification

    def matches(self) -> bool:
        return self.species == self.classification


class Hyperparameter:
    """하이퍼 파라미터 값과 분류의 전체 품질"""

    def __init__(self, k: int, training: "TrainingData") -> None:
        self.k = k
        self.data: weakref.ReferenceType["TrainingData"] = weakref.ref(training)
        self.quality: float

    def test(self) -> None:
        """전체 테스트 스위트 실행"""
        training_data: Optional["TrainingData"] = self.data()
        if not training_data:
            raise RuntimeError("Broken weak References")
        pass_count, fail_count = 0, 0
        for sample in training_data.testing:
            sample.classification = self.classify(sample)
            if sample.matches():
                pass_count += 1
            else:
                fail_count += 1
        self.quality = pass_count / (pass_count + fail_count)


class Purpose(enum.IntEnum):
    Classification = 0
    Testing = 1
    Training = 2


class TrainingData:
    """샘플을 로드하고 테스트하는 메소드를 가짐, 학습 및 테스트 데이터셋 포함"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.uploaded: datetime.datetime
        self.tested: datetime.datetime
        self.training: List[TrainingKnownSample] = []
        self.testing: List[TrainingKnownSample] = []
        self.tuning: List[Hyperparameter] = []

    def load(
        self,
        raw_data_iter: Iterable[dict[str, str]]
    ) -> None:
        """원시 데이터 로드 및 분할"""
        bad_count = 1
        for n, row in enumerate(raw_data_iter):
            try:
                if n % 5 == 0:
                    test = TestingKnownSample.from_dict(row)
                    self.testing.append(test)
                else:
                    train = TrainingKnownSample.from_dict(row)
                    self.testing.append(train)
            except InvalidSampleError as ex:
                print(f"Row {n+1}: {ex}")
                bad_count += 1
        if bad_count != 0:
            print(f"{bad_count} invalid rows")
            return
        self.uploaded = datetime.datetime.now(tz=datetime.timezone.utc)

    def test(
        self,
        parameter: Hyperparameter,
    ) -> None:
        """해당 파라미터 값으로 테스트"""
        parameter.test()
        self.tuning.append(parameter)
        self.tested = datetime.datetime.now(tz=datetime.timezone.utc)

    def classify(
        self,
        parameter: Hyperparameter,
        sample: Sample,
    ) -> Sample:
        """샘플 분류"""
        classification = parameter.classify(sample)
        sample.classifiy(classification)
        return sample

class Distance():
    def distance(self, s1: Sample, s2: Sample) -> float:
        pass


class ED(Distance):
    """유클리드 거리"""
    def distance(self, s1: Sample, s2: Sample) -> float:
        return math.hypot(
            s1.sepal_length - s2.sepal_length,
            s1.sepal_width - s2.sepal_width,
            s1.petal_length - s2.petal_length,
            s1.petal_width - s2.petal_width,
        )


class MD(Distance):
    """맨해튼 거리"""
    def distance(self, s1: Sample, s2: Sample) -> float:
        return sum(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),
            ]
        )


class CD(Distance):
    """체비쇼프 거리"""
    def distance(self, s1: Sample, s2: Sample) -> float:
        return max(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),
            ]
        )


class SD(Distance):
    """쇠렌센 거리"""
    def distance(self, s1: Sample, s2: Sample) -> float:
        return sum(
          [
              abs(s1.sepal_length - s2.sepal_length),
              abs(s1.sepal_width - s2.sepal_width),
              abs(s1.petal_length - s2.petal_length),
              abs(s1.petal_width - s2.petal_width),
          ]
      ) / sum(
          [
              s1.sepal_length + s2.sepal_length,
              s1.sepal_width + s2.sepal_width,
              s1.petal_length + s2.petal_length,
              s1.petal_width + s2.petal_width,
          ]
      )


class InvalidSampleError(ValueError):
    """소스 데이터 파일이 유효하지 않은 데이터 표현 사용"""
    pass

class BadSampleRow(ValueError):
    pass


class OutlierError(ValueError):
    """값이 예상된 범위 밖"""
    pass


class KnownSample(Sample):
    def __init__(
        self,
        species: str,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
        purpose: int,
    ) -> None:
        purpose_enum = Purpose(purpose)
        if purpose_enum not in {Purpose.Training, Purpose.Testing}:
            raise ValueError(f"Invalid purpose: {purpose!r}: {purpose_enum}")
        super().__init__(
            sepal_length = sepal_length,
            sepal_width = sepal_width,
            petal_length = petal_length,
            petal_width = petal_width,
        )
        self.purpose = purpose_enum
        self.species = species
        self._classification: Optional[str] = None

    def matches(self) -> bool:
        return self.species == self.classification

    def __repr__(self) -> str:
        return(
            f"{self.__class__.__name__}("
            f"sepal_length = {self.sepal_length}, "
            f"sepal_width = {self.sepal_width}, "
            f"petal_length = {self.petal_length}, "
            f"petal_width = {self.petal_width}, "
            f"sepecies = {self.species!r}, "
            f")"
        )

    @property
    def classification(self) -> Optional[str]:
        if self.purpose == Purpose.Training:
            return self._classification
        else:
            raise AttributeError(f"Training samples have no classification")

    @classification.setter
    def classification(self, value: str) -> None:
        if self.purpose == Purpose.Testing:
            self._classification = value
        else:
            raise AttributeError(f"Training samples cannot be classified")

    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "KnownSample":
        try:
            return cls(
                species = row["species"],
                sepal_length = float(row["sepal_length"]),
                sepal_width = float(row["sepal_width"]),
                petal_length = float(row["petal_length"]),
                petal_width = float(row["petal_width"]),
            )
        except ValueError as ex:
            raise InvalidSampleError(f"invalid {row!r}")


class TrainingKnownSample(KnownSample):
    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "TrainingKnownSample":
        return cast(TrainingKnownSample, super().from_dict(row))

class TestingKnownSample:
    pass


class SampleReader:
    target_class = Sample
    header = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

    def __init__(self, source: Path) -> None:
        self.source = source

    def sample_iter(self) -> Iterator[Sample]:
        target_class = self.target_class
        with self.source.open() as source_file:
            reader = csv.DictReader(source_file, self.header)
            for row in reader:
                try:
                    sample = target_class(
                        sepal_length = float(row["sepal_length"]),
                        sepal_width = float(row["sepal_width"]),
                        petal_length = float(row["petal_length"]),
                        petal_width = float(row["petal_width"]),
                    )
                except ValueError as ex:
                    raise BadSampleRow(f"Invalid {row!r}") from ex
                yield sample


class SampleDict(TypedDict):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    species: str



class SamplePartition(List[SampleDict], abc.ABC):
    @overload
    def __init__(self, *, training_subset: float = 0.80) -> None:
        ...

    @overload
    def __init__(self, iterable: Optional[Iterable[SampleDict]] = None, *, training_subset: float = 0.80) -> None:
        ...

    @overload
    def __init__(
        self,
        iterable: Optional[Iterable[SampleDict]] = None,
        *,
        training_subset: float = 0.80,
    ) -> None:
        self.training_subset = training_subset
        if iterable:
            super().__init__(iterable)
        else:
            super().__init__()

    @abc.abstractproperty
    @property
    def training(self) -> List[TrainingKnownSample]:
        ...

    @abc.abstractproperty
    @property
    def testing(self) -> List[TestingKnownSample]:
        ...


class ShufflingSamplePartition(SamplePartition):
    def __init__(
        self,
        iterable: Optional[Iterable[SampleDict]] = None,
        *,
        training_subset: float = 0.80,
    ) -> None:
        super().__init__(iterable, training_subset=training_subset)
        self.split: Optional[int] = None

    def shuffle(self) -> None:
        if not self.split:
            random.shuffle(self)
            self.split = int(len(self) * self.training_subset)

    @property
    def training(self) -> List[TrainingKnownSample]:
        self.shuffle()
        return [TrainingKnownSample(**sd) for sd in self[: self.split]]

    @property
    def testing(self) -> List[TestingKnownSample]:
        self.shuffle()
        return [TestingKnownSample(**sd) for sd in self[self.split :]]


class DealingPartition(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        items: Optional[Iterable[SampleDict]],
        *,
        training_subset: tuple[int, int] = (8, 10),
    ) -> None:
        ...

    @abc.abstractmethod
    def extend(self, items: Iterable[SampleDict]) -> None:
        ...

    @abc.abstractmethod
    def append(self, item: SampleDict) -> None:
        ...

    @property
    @abc.abstractmethod
    def training(self) -> List[TrainingKnownSample]:
        ...

    @property
    @abc.abstractmethod
    def testing(self) -> list[TestingKnownSample]:
        ...


class CountingDealingPartition(DealingPartition):
    def __init__(
        self,
        items: Optional[Iterable[SampleDict]],
        *,
        training_subset: tuple[int, int] = (8, 10),
    ) -> None:
        self.training_subset = training_subset
        self.counter = 0
        self._training: list[TrainingKnownSample] = []
        self._testing: list [TestingKnownSample] = []
        if items:
            self.extend(items)

    def extend(self, items: Iterable[SampleDict]) -> None:
        for item in items:
            self.append(item)

    def append(self, item: SampleDict) -> None:
        n, d = self.training_subset
        if self.counter % d < n:
            self._training.append(TrainingKnownSample(**item))
        else:
            self._testing.append(TestingKnownSample(**item))
        self.counter += 1

    @property
    def training(self) -> list[TrainingKnownSample]:
        return self._training

    @property
    def testing(self) -> list[TestingKnownSample]:
        return self._testing
