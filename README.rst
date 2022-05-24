daisybell
~~~~~~~~~~

A scanner that will scan your AI models for problems. Currently it focuses on bias testing. It is currently pre-alpha.


How to Use
~~~~~~~~~~

First install it:

::

    pip install daisybell


Run it in this manner (currently supports models from HuggingFace's repository):

::

    daisybell --huggingface roberta-base --task fill-mask


The scan can output files for further analysis:

::

    daisybell --huggingface roberta-base --task fill-mask --output results/roberta-base

Here is another example with a different bias task.

::

    daisybell --huggingface cross-encoder/nli-distilroberta-base --task zero-shot-classification

That's it for now. More will come.


Future Work
~~~~~~~~~~~~

* More bias tests. More metrics for bias testing based on the research in the field.
* Integration with other types of testing (eg. adversarial robustness)
* More kinds of models besides HuggingFace models. We are especially interested in MLFlow integration.
* Documentation.

Please contribute if you can. Help is always helpful.

License
~~~~~~~

Apache

Credit
~~~~~~

A project of IQT Labs.
