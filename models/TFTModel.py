
def make_rolling_window(data_test, firstDay: int, ndays: int, lag: int, horizon, best_tft, known_reals, unknown_reals,
                        group_ids, target):

    firstDay = firstDay * 24
    plot_horizon = 24 * ndays  # in hours

    x_ax = data_test['DateTime'][firstDay:firstDay + plot_horizon]

    preds_to_plot = []
    actuals_to_plot = data_test[firstDay + lag:firstDay + lag + plot_horizon][target]

    df_copy = data_test.copy()
    for i in range(plot_horizon // horizon):
        print(i)
        data_t = data_test[firstDay + (horizon) * i:firstDay + (horizon * (i)) + lag]
        test_dl = make_dataLoaders(horizon, lag, data_t, target, known_reals, unknown_reals, group_ids, train=False)
        raw_predictions = best_tft.predict(test_dl, mode='raw', return_x=False)
        preds_to_plot.append(raw_predictions['prediction'][0][:, 3].tolist())
        df_copy[firstDay + (horizon * (i)) + lag:firstDay + horizon * (i + 1) + lag][target] = \
        raw_predictions['prediction'][0][:, 3].tolist()

    final_pred_list = np.concatenate(preds_to_plot).ravel().tolist()  # flatten

    # convert to dataframe
    final_plot = actuals_to_plot.to_frame()
    final_plot.insert(loc=1, column='preds', value=final_pred_list)
    final_plot.insert(loc=0, column='time', value=x_ax)
    return final_plot





def make_dataLoaders(horizon, lag, data, target, known_reals, unknown_reals, group_ids, train=True):
    max_prediction_length = horizon
    max_encoder_length = lag

    training = TimeSeriesDataSet(
        data=data if train else None,
        time_idx="time_idx",
        target=target,
        group_ids=[group_ids],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=[group_ids],
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, data, predict=False, stop_randomization=True)

    batch_size = 32
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0) if train else None

    return training, train_dataloader, val_dataloader if train else val_dataloader


def setupTFT(training, max_epochs, n_gpus):
    # configure network and trainer
    pl.seed_everything(42)

    # configure network and trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("lightning_logs")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=n_gpus,
        weights_summary="top",
        gradient_clip_val=0.04925044813240137,
        limit_train_batches=30,
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.0230465326681773,
        hidden_size=10,
        attention_head_size=2,
        dropout=0.2982595590653191,
        hidden_continuous_size=42,
        output_size=7,  # 7 quantiles by default
        loss=QuantileLoss(),
        log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        reduce_on_plateau_patience=4,
    )
    print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")
    return trainer, tft


tf.io.gfile = tb.compat.tensorflow_stub.io.gfile  # Prevents some bugs, don't know why


def learnTFT(horizon, lag, data, target, known_reals, unknown_reals, group_ids, max_epochs, n_gpus=0):
    training, train_dataloader, val_dataloader = make_dataLoaders(horizon, lag, data, target, known_reals,
                                                                  unknown_reals, group_ids, train=True)
    trainer, tft = setupTFT(training, max_epochs, n_gpus)

    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    return best_tft


def predic_tft(data_test, known_reals, unknown_reals, group_ids, best_tft, target, firstDay=0, ndays=5, lag=18,
               horizon=6):
    final_df = make_rolling_window(data_test, firstDay, ndays, lag, horizon, best_tft, known_reals, unknown_reals,
                                   group_ids, target)
    return final_df
