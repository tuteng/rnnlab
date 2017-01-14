=====================================
Rnnlab
=====================================

Python API to train and analyze RNN language models using Tensorflow

Installation:
==============

Rnnlab can be installed via pypi:

``pip install rnnlab``

Before Using:
==============

In your your home directory, create a file 'rnnlab_user_configs.csv'. Inside, a number of required, and optional,
training hyperparameters and courpus information may be specified.

A bare-bones example of the configurations file for a single model:

+---------------+------------------+--------------+--------------+
| learning_rate | num_hidden_units | corpus_name  | probes_name  |
+---------------+------------------+--------------+--------------+
| 0.03          | 512              | 'childes'    | 'semantic'   |
+---------------+------------------+--------------+--------------+


Example Script:
================

This script imports ``gen_user_configs`` which loads the information contained in
the configurations file created above. In combination with a ``for`` loop, multiple
configurations can be loaded for sequential training of multiple models.

We also import the ``RNN`` class, which, when instantiated, creates a Tensorflow graph of the user-specified
RNN architecture. This class contains a ``train`` ing method which is used to train the model.

.. -code-begin-

.. code-block:: pycon

   >>> from rnnlab import gen_user_configs, get_childes_data
   >>> from rnnlab import RNN
   >>>
   >>> get_childes_data() # downloads childes corpus from github
   >>>
   >>> for user_configs in gen_user_configs():
   >>>     myrnn = RNN('lstm', user_configs) # try 'srn', ''irnn', 'scrn'
   >>>     myrnn.train()



During training, hidden state activations for user-specified words (probes) are saved into a pandas dataframe and saved
to disk. An included web application is used to visualize the data during and after training.

Project Information
===================

``rnnlab`` is released under the `MIT <http://choosealicense.com/licenses/mit/>`_ license,
the code on `GitHub <https://github.com/phueb/rnnlab>`_,
and the latest release on `PyPI <https://pypi.org/project/rnnlab/>`_.
Tested on Python 2.7.
