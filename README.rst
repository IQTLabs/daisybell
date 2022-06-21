daisybell
~~~~~~~~~~

A scanner that will scan your AI models for problems. Currently it focuses on bias testing. It is currently alpha.


How to Use
~~~~~~~~~~

First install it:

::

    pip install daisybell


Run it in this manner (currently supports models from HuggingFace's repository):

::

    daisybell roberta-base


The scan can output files for further analysis:

::

    daisybell roberta-base --output results/roberta-base

We will infer the task(s) of model by default but to provide specific tasks to test explicitly use the --task switch:

::

    daisybell cross-encoder/nli-distilroberta-base --task zero-shot-classification

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
