AI Scan
~~~~~~~

This is a scanner that will scan your AI models for problems. Currently it focuses on bias testing. It is currently pre-alpha.


How to Use
~~~~~~~~~~

First install it:

::

    pip install aiscan


Run it in this manner (currently supports models from HuggingFace's repository):

::

    aiscan --huggingface roberta


If it is a masking model, it will have output. That's it for now. More will come.


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
