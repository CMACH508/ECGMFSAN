import  math 
import  torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from Evaluation.test import test

def train(model,source1_loader=None,source2_loader=None,source3_loader=None,target_loader=None,weight=None,log_dir=None, optimizer=None, scheduler=None,epochs=15000,cuda=True,test_interval=50,writer=SummaryWriter()):
    # 最后的全连接层学习率为前面的10倍
    best_challengescore = 0.0
    source1_iter = iter(source1_loader)
    source2_iter = iter(source2_loader)
    source3_iter = iter(source3_loader)
    target_iter = iter(target_loader)

    correct = 0
    scale_factor = 1e-3
    total_cls_loss1 = 0
    total_cls_loss2 = 0
    total_cls_loss3 = 0


    total_mmd_loss1 = 0
    total_mmd_loss2 = 0
    total_mmd_loss3 = 0

    total_l1_loss1 = 0
    total_l1_loss2 = 0
    total_l1_loss3 = 0

    total_loss1 = 0
    total_loss2 = 0
    total_loss3 = 0

    for i in range(1, epochs + 1):
        model.train()

        try:
            source_data, source_label = source1_iter.next()
        except Exception as err:
            source1_iter = iter(source1_loader)
            source_data, source_label = source1_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, __ = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        # source_data, source_label = Variable(source_data), Variable(source_label)
        # target_data = Variable(target_data)
        optimizer.zero_grad()

        # cls_loss, mmd_loss, l1_loss = model(source_data, target_data, source_label, source_label, 1)
        cls_loss, mmd_loss, l1_loss, cls_loss4 = model(source_data, target_data, source_label, 1)

        gamma = 2 / (1 + math.exp(-10 * (i) / (epochs))) - 1
        gamma *= scale_factor
        loss = cls_loss4 + cls_loss + gamma * (mmd_loss + l1_loss)
        # loss = cls_loss + gamma * (mmd_loss + l1_loss)

        total_cls_loss1 += cls_loss.item()
        total_mmd_loss1 += mmd_loss.item()
        total_l1_loss1 += l1_loss.item()
        total_loss1 += loss.item()
        loss.backward()
        optimizer.step()

        if i % test_interval == 0:
            print(
                'Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                    i, 100. * i / epochs, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()))
            writer.add_scalar(tag='loss1', scalar_value=total_loss1)
            writer.add_scalar(tag='cls_loss1', scalar_value=total_cls_loss1)
            writer.add_scalar(tag='mmd_loss1', scalar_value=total_mmd_loss1)
            writer.add_scalar(tag='l1_loss1', scalar_value=total_l1_loss1)
            total_loss1 = 0
            total_cls_loss1 = 0
            total_mmd_loss1 = 0
            total_l1_loss1 = 0

        # if i % 3 == 2:
        try:
            source_data, source_label = source2_iter.next()
        except Exception as err:
            source2_iter = iter(source2_loader)
            source_data, source_label = source2_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, __ = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()

        optimizer.zero_grad()

        cls_loss, mmd_loss, l1_loss, cls_loss4 = model(source_data, target_data, source_label, 2)
        gamma = 2 / (1 + math.exp(-10 * (i) / (epochs))) - 1
        gamma *= scale_factor
        # loss = cls_loss + gamma * (mmd_loss + l1_loss)
        loss = cls_loss4 + cls_loss + gamma * (mmd_loss + l1_loss)

        total_cls_loss2 += cls_loss.item()
        total_mmd_loss2 += mmd_loss.item()
        total_l1_loss2 += l1_loss.item()
        total_loss2 += loss.item()

        loss.backward()
        optimizer.step()

        if i % test_interval == 0:
            # print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
            #     i, 100. * i / epochs, loss.data[0], cls_loss.data[0], mmd_loss.data[0], l1_loss.data[0]))
            print(
                'Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                    i, 100. * i / epochs, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()))
            writer.add_scalar(tag='loss2', scalar_value=total_loss2)
            writer.add_scalar(tag='cls_loss2', scalar_value=total_cls_loss2)
            writer.add_scalar(tag='mmd_loss2', scalar_value=total_mmd_loss2)
            writer.add_scalar(tag='l1_loss2', scalar_value=total_l1_loss2)
            total_loss2 = 0
            total_cls_loss2 = 0
            total_mmd_loss2 = 0
            total_l1_loss2 = 0

        # source3
        try:
            source_data, source_label = source3_iter.next()
        except Exception as err:
            source3_iter = iter(source3_loader)
            source_data, source_label = source3_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, __ = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        optimizer.zero_grad()

        # cls_loss, mmd_loss, l1_loss = model(source_data, target_data, source_label, source_label, 3)
        cls_loss, mmd_loss, l1_loss, cls_loss4 = model(source_data, target_data, source_label, 3)
        gamma = 2 / (1 + math.exp(-10 * (i) / (epochs))) - 1
        gamma *= scale_factor
        loss = cls_loss4 + cls_loss + gamma * (mmd_loss + l1_loss)
        # loss = cls_loss + gamma * (mmd_loss + l1_loss)
        total_cls_loss3 += cls_loss.item()
        total_mmd_loss3 += mmd_loss.item()
        total_l1_loss3 += l1_loss.item()
        total_loss3 += loss.item()
        loss.backward()
        optimizer.step()

        if i % test_interval == 0:
            # print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
            #     i, 100. * i / epochs, loss.data[0], cls_loss.data[0], mmd_loss.data[0], l1_loss.data[0]))
            print(
                'Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                    i, 100. * i / epochs, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()))
            writer.add_scalar(tag='loss3', scalar_value=total_loss3)
            writer.add_scalar(tag='cls_loss3', scalar_value=total_cls_loss3)
            writer.add_scalar(tag='mmd_loss3', scalar_value=total_mmd_loss3)
            writer.add_scalar(tag='l1_loss3', scalar_value=total_l1_loss3)
            total_loss3 = 0
            total_cls_loss3 = 0
            total_mmd_loss3 = 0
            total_l1_loss3 = 0

        if i % (test_interval) == 0:
            model.eval()
            # if i % ( 1) == 0:
            challenge_score = test(model,target_loader=target_loader,weight=weight,cuda=cuda,writer=writer)
            if challenge_score > best_challengescore:
                best_challengescore = challenge_score
                torch.save(model, log_dir + 'best.pth')
            model.train()
            if scheduler is not None:
                scheduler.step()