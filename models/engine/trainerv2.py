import torch
import time
from my_utils.ml_acc import calculate_accuracy_mode_one
import torch.nn.functional as F


def do_train_single_label_cpu(model, optimizer: torch.optim.Optimizer,
                              loss_fn: torch.nn.modules.loss._Loss,
                              train_loader, valid_data,
                              vis, max_epochs, scheduler, plot_name):
    counter = 0
    t0 = time.time()
    model.train()

    for epoch in range(max_epochs):
        for epi, (bx, by) in enumerate(train_loader):
            # -- training --------------------------------------------------
            logits = model(bx)
            training_loss = loss_fn(logits, by)
            training_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (epi + 1) % 2 == 0 and vis is not None:
                bx_ind, by_ind = valid_data
                model.eval()
                with torch.no_grad():
                    logits_ind = model(bx_ind)
                    validation_loss = loss_fn(logits_ind, by_ind)
                model.train()
                # scheduler.step(validation_loss)  # reduce scheduler

                acc_ind = (logits_ind.argmax(1) == by_ind).float().mean().item()
                acc_tra = (logits.argmax(1) == by).float().mean().item()

                vis.line(Y=[[acc_tra, acc_ind]], X=[counter],
                         update=None if counter == 0 else 'append', win=f'Acc_{plot_name}',
                         opts=dict(legend=['train', 'val'], title=f'Acc_{plot_name}'))
                # vis.line(Y=[[training_loss.item(), validation_loss.item()]], X=[counter],
                #          update=None if counter == 0 else 'append', win=f'Loss_{self.trail_name}',
                #          opts=dict(legend=['train', 'val'], title=f'Loss_{self.trail_name}'))
                counter += 1

        scheduler.step()  # exp scheduler

        if (epoch + 1) % 10 == 0:
            bx_ind, by_ind = valid_data
            model.eval()
            with torch.no_grad():
                logits_ind = model(bx_ind)
            model.train()
            acc_ind = (logits_ind.argmax(1) == by_ind).float().mean().item()
            print(f"[Epoch-{epoch + 1}/{max_epochs}] Test Acc. {acc_ind:.2%}, {time.time() - t0:.2f}s from Ep-1")

    t01 = time.time()
    bx_ind, by_ind = valid_data
    model.eval()
    with torch.no_grad():
        logits_ind = model(bx_ind)
    model.train()
    t02 = time.time()

    acc_ind = (logits_ind.argmax(1) == by_ind).float().mean().item()
    print(f"\n[{plot_name}] Test Acc. {acc_ind:.2%}")
    results = {'test_acc': acc_ind, 'test_time': t02 - t01}

    return results


def do_train_multilabel_cpu(model, optimizer: torch.optim.Optimizer,
                            loss_fn: torch.nn.modules.loss._Loss,
                            train_loader, valid_data,
                            vis, max_epochs, scheduler, plot_name,
                            num_cls, num_cls_train):
    counter = 0
    # sigmoid_fun = torch.nn.Sigmoid()
    bx_ind, by_ind = valid_data
    by_ind = by_ind.float()
    t0 = time.time()
    model.train()

    for epoch in range(max_epochs):
        for epi, (bx, by) in enumerate(train_loader):
            by = by.float()

            # -- training --------------------------------------------------
            logits = model(bx)
            training_loss = loss_fn(logits, by)
            training_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (epi + 1) % 2 == 0 and vis is not None:
                model.eval()
                with torch.no_grad():
                    logits_ind = model(bx_ind)
                    # validation_loss = loss_fn(logits_ind, by_ind)
                model.train()
                # scheduler.step(validation_loss)

                # acc_ind = (logits_ind.argmax(1) == by_ind).float().mean().item()
                # acc_tra = (logits.argmax(1) == by).float().mean().item()

                val_p, val_r, val_acc, val_acc_sc = calculate_accuracy_mode_one(logits_ind.sigmoid(), by_ind,
                                                                               False, num_cls, num_cls_train)
                tra_p, tra_r, tra_acc, tra_acc_sc = calculate_accuracy_mode_one(logits.sigmoid(), by, True)

                vis.line(Y=[[tra_p, val_p]], X=[counter],
                         update=None if counter == 0 else 'append', win=f'ML_Precision_{plot_name}',
                         opts=dict(legend=['train', 'valid'], title=f'ML_Precision_{plot_name}'))
                vis.line(Y=[[tra_r, val_r]], X=[counter],
                         update=None if counter == 0 else 'append', win=f'ML_Recall_{plot_name}',
                         opts=dict(legend=['train', 'valid'], title=f'ML_Recall_{plot_name}'))
                vis.line(Y=[[tra_acc, val_acc]], X=[counter],
                         update=None if counter == 0 else 'append', win=f'ML_Acc_{plot_name}',
                         opts=dict(legend=['train', 'valid'], title=f'ML_Acc_{plot_name}'))
                counter += 1
        scheduler.step()  # exp scheduler

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits_ind = model(bx_ind).sigmoid()
            model.train()
            val_p, val_r, val_acc, val_acc_sc = calculate_accuracy_mode_one(
                logits_ind, by_ind, False, num_cls, num_cls_train)
            print("[Epoch-{}/{}] Test precision: {:.2%}, recall: {:.2%}, f1-score: {:.2%}, "
                  "acc: {:.2%}, val_acc_sc: {:.2%}, {:.2f} from Ep-1".format(
                epoch + 1, max_epochs, val_p, val_r, 2 * val_p * val_r / (val_p + val_r),
                val_acc, val_acc_sc, time.time() - t0))

    t01 = time.time()
    model.eval()
    with torch.no_grad():
        logits_ind = model(bx_ind).sigmoid()
    model.train()
    t02 = time.time()

    val_p, val_r, val_acc, val_acc_sc = calculate_accuracy_mode_one(logits_ind, by_ind, False, num_cls, num_cls_train)
    print("\n[{}] Test precision: {:.2%}, recall: {:.2%}, f1-score: {:.2%}, acc: {:.2%}, acc_sc: {:.2%}".format(
        plot_name, val_p, val_r, 2 * val_p * val_r / (val_p + val_r), val_acc, val_acc_sc))

    results = {'val_precision': val_p, 'val_recall': val_r, 'f1-score': 2 * val_p * val_r / (val_p + val_r),
               'val_acc_ml': val_acc, 'val_acc_sc': val_acc_sc, 'test_time': t02 - t01}
    return results

# ------------------- GPU -----------------------------
# ------------------- GPU -----------------------------


def do_train_single_label_gpu(model, optimizer: torch.optim.Optimizer,
                              loss_fn: torch.nn.modules.loss._Loss,
                              train_loader, valid_loader, valid_loader_inf,
                              vis, max_epochs, scheduler, plot_name, samples=None):
    model = model.cuda()
    counter = 0
    valid_loader_inf = iter(valid_loader_inf)
    # t0 = time.time()
    model.train()

    for epoch in range(max_epochs):
        # if isinstance(model, nn.Module):
        #     if hasattr(model, 'plot_OFR') and (epoch + 1) % 5 == 0:
        #         model.plot_OFR(samples)

        for epi, (bx, by) in enumerate(train_loader):
            bx, by = bx.cuda().float(), by.cuda().long()

            # -- training --------------------------------------------------
            # print('bx:', bx.shape)
            logits = model(bx)
            training_loss = loss_fn(logits, by)
            training_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (epi + 1) % 2 == 0 and vis is not None:
                bx_ind, by_ind = valid_loader_inf.__next__()
                bx_ind, by_ind = bx_ind.cuda(), by_ind.cuda().long()
                model.eval()
                with torch.no_grad():
                    logits_ind = model(bx_ind)
                    validation_loss = loss_fn(logits_ind, by_ind)
                model.train()
                # scheduler.step(validation_loss)  # reduce scheduler

                acc_ind = (logits_ind.argmax(1) == by_ind).float().mean().item()
                acc_tra = (logits.argmax(1) == by).float().mean().item()

                vis.line(Y=[[acc_tra, acc_ind]], X=[counter],
                         update=None if counter == 0 else 'append', win=f'Acc_{plot_name}',
                         opts=dict(legend=['train', 'val'], title=f'Acc_{plot_name}'))
                vis.line(Y=[[training_loss.item(), validation_loss.item()]], X=[counter],
                         update=None if counter == 0 else 'append', win=f'Loss_{plot_name}',
                         opts=dict(legend=['train', 'val'], title=f'Loss_{plot_name}'))
                counter += 1

        scheduler.step()  # exp scheduler

        if (epoch + 1) % 10 == 0:
            model.eval()
            test_num = 0
            pred_result = 0.
            with torch.no_grad():
                for bx, by in valid_loader:
                    bx, by = bx.cuda(), by.cuda().long()
                    _logits = model(bx)
                    pred_result += (_logits.argmax(1) == by).sum()
                    test_num += len(by)
            model.train()
            acc = pred_result / test_num
            print(f"[Epoch-{epoch + 1}/{max_epochs}] Test Acc. {acc.item():.2%}")

    # test on gpu
    model.eval()
    test_num = 0
    pred_result = 0.
    t0 = time.time()
    with torch.no_grad():
        for bx, by in valid_loader:
            bx, by = bx.cuda(), by.cuda().long()
            _logits = model(bx)
            pred_result += (_logits.argmax(1) == by).sum()
            test_num += len(by)
    model.train()
    t1 = time.time()
    acc = pred_result / test_num
    print(f"\n[{plot_name}] Test Acc. {acc.item():.2%}")

    # test on cpu:
    model = model.to('cpu')
    t2 = time.time()
    test_num = 0
    pred_result = 0.
    with torch.no_grad():
        for bx, by in valid_loader:
            bx, by = bx.float(), by.long()
            _logits = model(bx)
            pred_result += (_logits.argmax(1) == by).sum()
            test_num += len(by)
    model.train()
    _ = pred_result / test_num
    t3 = time.time()

    model = model.cuda()
    print(f"\n[{plot_name}] Test Acc. {acc.item():.2%}")
    results = {'test_acc': acc.item(), 'test_time': t1 - t0, 'test_time_cpu': t3 - t2}
    print(f"test_time_cpu: {t3 - t2: .3f}")
    return results


def do_train_multilabel_gpu(model, optimizer: torch.optim.Optimizer,
                            loss_fn: torch.nn.modules.loss._Loss,
                            train_loader, valid_loader, valid_loader_inf,
                            vis, max_epochs, scheduler, plot_name,
                            num_cls, num_cls_train):
    model = model.cuda()
    counter = 0
    valid_loader_inf = iter(valid_loader_inf)
    model.train()

    for epoch in range(max_epochs):
        for epi, (bx, by) in enumerate(train_loader):
            # -- (optional) minimal lr search ------------------------------
            # not implemented
            bx, by = bx.cuda(), by.cuda().float()

            # -- training --------------------------------------------------
            logits = model(bx)
            training_loss = loss_fn(logits, by)
            training_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (epi + 1) % 2 == 0 and vis is not None:
                bx_ind, by_ind = valid_loader_inf.__next__()
                bx_ind, by_ind = bx_ind.cuda(), by_ind.cuda().float()

                model.eval()
                with torch.no_grad():
                    logits_ind = model(bx_ind)
                    validation_loss = loss_fn(logits_ind, by_ind)
                model.train()
                # scheduler.step(validation_loss)

                # acc_ind = (logits_ind.argmax(1) == by_ind).float().mean().item()
                # acc_tra = (logits.argmax(1) == by).float().mean().item()

                val_p, val_r, val_acc, val_acc_sc = calculate_accuracy_mode_one(
                    logits_ind.sigmoid(), by_ind, False, num_cls, num_cls_train)
                tra_p, tra_r, tra_acc, tra_acc_sc = calculate_accuracy_mode_one(logits.sigmoid(), by, True)

                vis.line(Y=[[tra_p, val_p]], X=[counter],
                         update=None if counter == 0 else 'append', win=f'ML_Precision_{plot_name}',
                         opts=dict(legend=['train', 'valid'], title=f'ML_Precision_{plot_name}'))
                vis.line(Y=[[tra_r, val_r]], X=[counter],
                         update=None if counter == 0 else 'append', win=f'ML_Recall_{plot_name}',
                         opts=dict(legend=['train', 'valid'], title=f'ML_Recall_{plot_name}'))
                vis.line(Y=[[tra_acc, val_acc]], X=[counter],
                         update=None if counter == 0 else 'append', win=f'ML_Acc_{plot_name}',
                         opts=dict(legend=['train', 'valid'], title=f'ML_Acc_{plot_name}'))
                counter += 1

        scheduler.step()  # exp scheduler

        if (epoch + 1) % 10 == 0:
            model.eval()
            val_logits = []
            val_labels = []
            with torch.no_grad():
                for bx, by in valid_loader:
                    bx, by = bx.cuda(), by.cuda().float()
                    _logits = model(bx)
                    val_logits.append(_logits)
                    val_labels.append(by)
            model.train()

            val_logits = torch.cat(val_logits, 0).sigmoid()
            val_labels = torch.cat(val_labels, 0)
            val_p, val_r, val_acc, val_acc_sc = calculate_accuracy_mode_one(val_logits, val_labels, False, num_cls,
                                                                           num_cls_train)
            print("[Epoch-{}/{}] Test precision: {:.2%}, recall: {:.2%}, f1-score: {:.2%}, "
                  "acc: {:.2%}, acc_sc: {:.2%}".format(
                epoch + 1, max_epochs, val_p, val_r, 2 * val_p * val_r / (val_p + val_r),
                val_acc, val_acc_sc))

    # test on gpu:
    model.eval()
    val_logits = []
    val_labels = []
    t0 = time.time()
    with torch.no_grad():
        for bx, by in valid_loader:
            bx, by = bx.cuda(), by.cuda().float()
            _logits = model(bx)
            val_logits.append(_logits)
            val_labels.append(by)
            # pred_result += (_logits.argmax(1) == by).sum()
            # test_num += len(by)
    model.train()
    t1 = time.time()

    val_logits = torch.cat(val_logits, 0).sigmoid()
    val_labels = torch.cat(val_labels, 0)
    val_p, val_r, val_acc, val_acc_sc = calculate_accuracy_mode_one(
        val_logits, val_labels, False, num_cls, num_cls_train)
    print(
        "\n[{}] Test precision: {:.2%}, recall: {:.2%}, f1-score: {:.2%}, "
        "acc: {:.2%}, acc_sc: {:.2%}, {:.2f}s from Ep-1".format(
            plot_name, val_p, val_r, 2 * val_p * val_r / (val_p + val_r), val_acc, val_acc_sc, time.time() - t0))
    # probs_val = torch.softmax(logits_val, 1)
    # self.reliability_plot_model(probs_val, Y_val)

    # test on cpu:
    model = model.to('cpu')
    t2 = time.time()
    model.eval()
    val_logits = []
    val_labels = []
    with torch.no_grad():
        for bx, by in valid_loader:
            bx, by = bx.float(), by.float()
            _logits = model(bx)
            val_logits.append(_logits)
            val_labels.append(by)
    model.train()
    val_logits = torch.cat(val_logits, 0).sigmoid()
    val_labels = torch.cat(val_labels, 0)
    _, _, _, _ = calculate_accuracy_mode_one(
        val_logits, val_labels, False, num_cls, num_cls_train)
    t3 = time.time()

    model = model.cuda()
    results = {'val_precision': val_p, 'val_recall': val_r, 'f1-score': 2 * val_p * val_r / (val_p + val_r),
               'val_acc_ml': val_acc, 'val_acc_sc': val_acc_sc, 'test_time': t1 - t0, 'test_time_cpu': t3 - t2}
    print(f"test_time_cpu: {t3 - t2: .3f}")
    return results


def do_train_multilabel_gpu_caps(model, optimizer: torch.optim.Optimizer,
                            loss_fn, train_loader, valid_loader, valid_loader_inf,
                            vis, max_epochs, scheduler, plot_name, num_cls):
    model = model.cuda()
    counter = 0
    valid_loader_inf = iter(valid_loader_inf)
    model.train()

    for epoch in range(max_epochs):
        for epi, (bx, by) in enumerate(train_loader):
            # -- training --------------------------------------------------
            bx, by = bx.cuda(), by.cuda().long()
            by = F.one_hot(by, num_cls)
            y_pred, x_recon = model(bx)
            training_loss = loss_fn(y_pred, by, bx, x_recon)
            training_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (epi + 1) % 2 == 0 and vis is not None:
                bx_val, by_val = valid_loader_inf.__next__()
                bx_val, by_val = bx_val.cuda(), by_val.cuda().long()
                by_val = F.one_hot(by_val, num_cls)

                model.eval()
                with torch.no_grad():
                    y_pred_val, x_recon_val = model(bx_val)
                    val_loss = loss_fn(y_pred_val, by_val, bx_val, x_recon_val)
                model.train()
                # scheduler.step(validation_loss)

                tra_acc = (y_pred.argmax(1) == by.argmax(1)).float().mean().item()
                val_acc = (y_pred_val.argmax(1) == by_val.argmax(1)).float().mean().item()

                vis.line(Y=[[tra_acc, val_acc]], X=[counter],
                         update=None if counter == 0 else 'append', win=f'Acc_{plot_name}',
                         opts=dict(legend=['train', 'valid'], title=f'Acc_{plot_name}'))
                vis.line(Y=[[training_loss.item(), val_loss.item()]], X=[counter],
                         update=None if counter == 0 else 'append', win=f'Loss_{plot_name}',
                         opts=dict(legend=['train', 'valid'], title=f'Loss_{plot_name}'))
                counter += 1

        scheduler.step()  # exp scheduler

        if (epoch + 1) % 10 == 0:
            model.eval()
            val_logits = []
            val_labels = []
            with torch.no_grad():
                for bx, by in valid_loader:
                    bx, by = bx.cuda(), by.cuda().long()
                    by = F.one_hot(by, num_cls)
                    y_pred_val, x_recon_val = model(bx)
                    val_logits.append(y_pred_val)
                    val_labels.append(by)
            model.train()

            val_preds = torch.cat(val_logits, 0)
            val_labels = torch.cat(val_labels, 0)
            val_acc_sc = (val_preds.argmax(1) == val_labels.argmax(1)).float().mean().item()
            print("[Epoch-{}/{}] Test  acc_sc: {:.2%}".format(epoch + 1, max_epochs, val_acc_sc))

    # test on gpu:
    t0 = time.time()
    model.eval()
    val_logits = []
    val_labels = []
    with torch.no_grad():
        for bx, by in valid_loader:
            bx, by = bx.cuda(), by.cuda().long()
            by = F.one_hot(by, num_cls)
            y_pred_val, x_recon = model(bx)
            val_logits.append(y_pred_val)
            val_labels.append(by)
    model.train()
    t1 = time.time()

    val_preds = torch.cat(val_logits, 0)
    val_labels = torch.cat(val_labels, 0)
    val_acc_sc = (val_preds.argmax(1) == val_labels.argmax(1)).float().mean().item()
    print(
        "\n[{}] Test acc_sc: {:.2%}, {:.2f}s from Ep-1".format(plot_name, val_acc_sc, time.time() - t0))
    # probs_val = torch.softmax(logits_val, 1)
    # self.reliability_plot_model(probs_val, Y_val)

    # test on cpu:
    model = model.to('cpu')
    t2 = time.time()
    model.eval()
    val_logits = []
    val_labels = []
    with torch.no_grad():
        for bx, by in valid_loader:
            bx, by = bx.float(), by.long()
            by = F.one_hot(by, num_cls)
            val_y_pred, x_recon = model(bx)
            val_logits.append(val_y_pred)
            val_labels.append(by)
    model.train()
    val_logits = torch.cat(val_logits, 0)
    val_labels = torch.cat(val_labels, 0)
    t3 = time.time()

    model = model.cuda()
    results = {'val_acc_sc': val_acc_sc, 'test_time': t1 - t0, 'test_time_cpu': t3 - t2}
    print(f"test_time_cpu: {t3 - t2: .3f}")
    return results