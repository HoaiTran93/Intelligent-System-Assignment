
run_config = { 'experiment_id' : "power_apr26",
               'data_folder': 'resources/data/discords/dutch_power/',
               'save_figure': True
               }

multi_step_lstm_config = {  'batch_size': 672,
                            'n_epochs': 600,
                            'dropout': 0.2,
                            'look_back': 1,
                            'look_ahead':1,
                            'layers': {'input': 1, 'hidden1': 300, 'output': 1},
                            'loss': 'mse',
                            'train_test_ratio' : 0.7,
                            'shuffle': False,
                            'validation': True,
                            'learning_rate': .01,
                            'patience':5,
                           }
