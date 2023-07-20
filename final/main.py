import CHF_model_api as CHF

if __name__ == '__main__':

    hparams = {
        'data_seed': 1,
        'name': None,
        'weight_seed': 1,
    }

    test = CHF.My_model(hparams)
    test.create()

    print(CHF.My_model.DATA)

    

