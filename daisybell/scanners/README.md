# Daisybell Scanners
All scanners are implemented as plugins to the Daisybell system. The system maintains a registry of known local plugins and upon being called against a given [Pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines) will iterate across all registered scanners. If a scanner is capable of scanning a given pipeline (typically based on it's type or task) then that scanner's `scan` method will be invoked and the results of said scan will be returned.

# Current Scanners
## MaskingLanguageBias
**Kind:** bias
**Tasks:** fill-mask
**Description:**
Scans for language bias in NLP masking models.

## NerLanguageBias
**Kind:** bias
**Tasks:** token-classification
**Description:**
Scans for language bias in NER based models. WARNING! THIS SCANNER IS EXPERIMENTAL.

## ZeroShotLanguageBias
**Kind:** bias
**Tasks:** zero-shot-classification
**Description:**
Scans for language bias in NLP zero shot models.

# Adding new Scanners
To add a new scanner create a class that inherits from `daisybell.scanners.ScannerBase`. We suggest adding it in the `daisybell/scanners` directory in order to make it easier to track however the location shouldn't matter. Loading the class into memory will automatically add the new scanner to the ScannerRegistry. For examples please look to the classes within the `daisybell.scanners` module. To execute properly your scanner class must implement the following methods:
### `__init__`
**Parameters:** None
**Returns:** None
This method will initialize any values that need to be set for the class to execut a successful scan. The first line of this scan must be:
`super().__init__(<NAME>, <KIND>, <DESCRIPTION>, logger)`
If this call is not made then the scanner will not be added to the `ScannerRegistry` and will not be called by subsequent `daisybell` scans.

### `can_scan`
**Parameters:**
pipeline: an instance of transformers.Pipeline
**Returns:** bool
This method should return True if the `scan` method is applicable to the supplied Pipeline, otherwise False

### `scan`
**Parameters:**
pipeline: an instance of transformers.Pipeline
params: a ditionary of parameters to be supplied to the pipeline
**Returns:** a pandas DataFrame
This method should encapsulate the desired test routine(s) to be run against the supplied pipeline and evaluated using the supplied params.
