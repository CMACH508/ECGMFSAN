import  torch
from torch.utils.tensorboard import SummaryWriter
from Evaluation.evaluation import compute_auc,compute_f_measure,get_challenge_score


def calMean(pred1,pred2,pred3,pred4,weight,alpha=0.40):
    return (1-alpha)*(weight[0][:]*pred1+weight[1][:]*pred2+weight[2][:]*pred3)+alpha*pred4


@torch.no_grad()
def test(model,target_loader,weight,cuda=True,writer=SummaryWriter()):
    # model.eval()
    print('True')

    thres1, thres2, thres3, thres4 = 0.5, 0.5, 0.5, 0.5
    for i, (data, target) in enumerate(target_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        pred1, pred2, pred3, pred4 = model(data)

        if i == 0:
            All_pred1 = pred1
            All_pred2 = pred2
            All_pred3 = pred3
            All_pred4 = pred4

            # All_pred_mean = (pred1 + pred2 + pred3+pred4) / 4
            All_pred_mean = calMean(pred1, pred2, pred3, pred4,weight)

            All_label = target
        else:
            All_pred1 = torch.cat([All_pred1, pred1], dim=0)
            All_pred2 = torch.cat([All_pred2, pred2], dim=0)
            All_pred3 = torch.cat([All_pred3, pred3], dim=0)
            All_pred4 = torch.cat([All_pred4, pred4], dim=0)
            # All_pred_mean = torch.cat([All_pred_mean, (pred1 + pred2 + pred3+pred4) / 4], dim=0)
            All_pred_mean = torch.cat([All_pred_mean, calMean(pred1, pred2, pred3, pred4,weight)], dim=0)
            All_label = torch.cat([All_label, target], dim=0)

    binary_outputs1 = (All_pred1.detach().cpu().numpy() >= thres1).astype('int')
    f1_score_1 = compute_f_measure(All_label.detach().cpu().numpy(), binary_outputs1)
    print('f1 score by source 1:', f1_score_1.mean())
    writer.add_scalar(tag='F1Score_1', scalar_value=f1_score_1)
    score_1 = get_challenge_score(All_label.detach().cpu().numpy(), All_pred1.detach().cpu().numpy())
    print('score by source 1:', score_1)
    writer.add_scalar(tag='score_1', scalar_value=score_1)

    del All_pred1

    binary_outputs2 = (All_pred2.detach().cpu().numpy() >= thres2).astype('int')
    f1_score_2 = compute_f_measure(All_label.detach().cpu().numpy(), binary_outputs2)
    print('f1 score by source 2:', f1_score_2.mean())
    writer.add_scalar(tag='F1Score_2', scalar_value=f1_score_2)

    score_2 = get_challenge_score(All_label.detach().cpu().numpy(), All_pred2.detach().cpu().numpy())
    print('score by source 2:', score_2)
    writer.add_scalar(tag='score_2', scalar_value=score_2)
    del All_pred2

    binary_outputs3 = (All_pred3.detach().cpu().numpy() >= thres3).astype('int')
    f1_score_3 = compute_f_measure(All_label.detach().cpu().numpy(), binary_outputs3)
    writer.add_scalar(tag='F1Score_3', scalar_value=f1_score_3)
    print('f1 score by source 3:', f1_score_3.mean())
    score_3 = get_challenge_score(All_label.detach().cpu().numpy(), All_pred3.detach().cpu().numpy())
    print('score by source 3:', score_3)
    writer.add_scalar(tag='score_3', scalar_value=score_3)
    del All_pred3

    binary_outputs4 = (All_pred4.detach().cpu().numpy() >= thres4).astype('int')
    f1_score_4 = compute_f_measure(All_label.detach().cpu().numpy(), binary_outputs4)
    writer.add_scalar(tag='F1Score_4', scalar_value=f1_score_4)
    print('f1 score by source 4:', f1_score_4.mean())
    score_4 = get_challenge_score(All_label.detach().cpu().numpy(), All_pred4.detach().cpu().numpy())
    print('score by source 4:', score_4)
    writer.add_scalar(tag='score_4', scalar_value=score_4)
    del All_pred4

    # majority voting
    # binary_outputs_mean = ((binary_outputs1 + binary_outputs2 + binary_outputs3+binary_outputs4) >= 2).astype('int')
    # print('sum:', sum(binary_outputs_mean))
    # print(binary_outputs_mean.shape, All_label.detach().cpu().numpy().shape)

    # Mean Mix
    binary_outputs_mean = (All_pred_mean.detach().cpu().numpy() > 0.5)
    f1_score_mean = compute_f_measure(All_label.detach().cpu().numpy(), binary_outputs_mean)
    print('f1 score by MSFAN:', f1_score_mean.mean())
    writer.add_scalar(tag='F1Score_MSFAN', scalar_value=f1_score_mean)

    auroc, auprc = compute_auc(All_label.detach().cpu().numpy(), All_pred_mean.detach().cpu().numpy())
    print('auroc:%.2f  auprc:%.2f' % (auroc, auprc))
    acc = torch.sum(torch.all(All_label.bool() == (All_pred_mean > 0.5), dim=1).int()) / All_label.shape[0]
    print('acc:', acc)
    score_MFSAN = get_challenge_score(All_label.detach().cpu().numpy(), All_pred_mean.detach().cpu().numpy())
    print('score by source _mean:', score_MFSAN)
    writer.add_scalar(tag='score_MFSAN', scalar_value=score_MFSAN)
    print('')
    del All_pred_mean

    del All_label
    return score_MFSAN