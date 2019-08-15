
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import torchvision.models
import random
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN

device = torch.device("cuda:0")

classes = ['Pear',
 'Orange',
 'Salad',
 'Grape',
 'Muffin',
 'Egg (Food)',
 'Banana',
 'Bread',
 'Cucumber',
 'Broccoli',
 'Cookie',
 'Carrot',
 'Cheese',
 'Strawberry',
 'Hot dog',
 'Bagel',
 'Lemon',
 'Apple',
 'Burrito',
 'Coffee',
 'French fries',
 'Pizza',
 'Juice',
 'Tomato',
 'Hamburger',
 'Pasta',
 'Waffle',
 'Potato',
 'Sandwich',
 'Doughnut',
 'Pancake',
 'Croissant']

def get_train_accuracy_mul_gpu(model,box_labels,img):
    model = model.eval()
    out = model(img)
    total = 0
    correct = 0
    for i in range(len(out)):
        if out[i]['labels'].tolist()!=[]:
            correct += out[i]['labels'][0].to(device)==box_labels[i]['labels'][0]
    else:
        total += 1
    model = model.train()
    return correct/total

def get_val_loss_acc_mul_gpu(val_model,batch_size=1):
    val_model = val_model.train()

    s_losses = []
    val_idx = 0

    correct = 0
    total = 0
    with torch.no_grad():
        for single in classes:
            imported = torch.load('./tensor/validation/'+single+'.pt')
            for info in imported:
                img = info[0].to(device)
                info[1]['boxes'] = info[1]['boxes']*640

                box_label = info[1]
                #box_label['boxes'] = box_label['boxes']*640

                box_cuda = {
                        "boxes" : box_label['boxes'].to(device),
                        "labels" : box_label['labels'].to(device)
                }
                loss_dict = val_model([img.to(device)],[box_cuda])
                losses = sum(loss for loss in loss_dict.values())
                s_losses.append(losses/batch_size)
                del loss_dict
                val_model = val_model.eval()
                if val_model([img.to(device)])[0]['labels'].tolist()!=[]:
                    out = val_model([img.to(device)])[0]['labels'][0]
                    if out==labels:
                        correct += 1
                    del out
                total += 1
                print (('\tprocessing iterration {}... Val Loss: {} | Val Acc: {}').format(val_idx,s_losses[-1],correct/total))
                val_idx += 1


                val_model = val_model.train()

    return [sum(s_losses)/len(s_losses),correct / total]

def train_net_mul_gpu(model, batch_size=4, num_epochs=100, learning_rate = 0.0006,weight_decay=0.0002):
    checkpoint_num = 0
    model = model.train()
    start_time = time.time()

    torch.manual_seed(1000)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)


    saved_x, saved_train_losses, saved_val_losses, train_acc, val_acc = [], [], [], [], []

    # training


    saved_idx = 0
    iter_acc = []
    iter_loss = []
    for epoch in range(num_epochs):
        #idx = 0
        #import one class
        total_list = os.listdir('./train_data/')
        while 1:
            if total_list==[]:
                break
            current_file = torch.load('./train_data/'+total_list[0])
            total_list = total_list[1:]
            for element in current_file:
                batch_img = []
                batch_box = []
                for j in element:
                    img = j[0].to(device)
                    j[1]['boxes'] = j[1]['boxes']*640

                    box_label = j[1]
                    #box_label['boxes'] = box_label['boxes']*640

                    box_cuda = {
                      "boxes" : box_label['boxes'].to(device),
                      "labels" : box_label['labels'].to(device)
                    }
                    batch_img.append(img)
                    batch_box.append(box_cuda)
                print('img_len:'+str(len(batch_img)))
                print('box_len:'+str(len(batch_box)))

                loss_dict = model(batch_img,batch_box)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                iter_acc.append(get_train_accuracy_one_gpu(model,batch_box,batch_img))
                iter_loss.append(float(losses)/batch_size)
                print (('{} processing iterration {}... Train Loss: {} | Train Acc: {}').format(current_file[:-3],i,iter_loss[-1],iter_acc[-1]))

            saved_x.append(saved_idx)
            saved_idx += 1

            saved_train_losses.append(sum(iter_loss)/len(iter_loss))
            [current_val_loss,current_val_acc] = get_val_loss_acc_one_gpu(model.state_dict())
            saved_val_losses.append(current_val_loss)
            print ('-------------------------------------------------------------------------------------------')
            #print(('Train loss: {} | Validation loss: {}').format(saved_train_losses[-1],saved_val_losses[-1]))
            train_acc.append(sum(iter_acc)/len(iter_acc)) # compute training accuracy
            iter_acc = []
            iter_loss = []
            val_acc.append(current_val_acc)  # compute validation accuracy

            print(("Epoch {}: Train acc: {} |"+"Validation acc: {}").format(epoch + 1,train_acc[-1],val_acc[-1]))
            #print(("Epoch {}: Train acc: {} ").format(epoch + 1,train_acc[-1]))
            print ('-------------------------------------------------------------------------------------------')
            model_path = "bs{0}_lr{1}_epoch{2}_checkpoint_{3}".format(batch_size,learning_rate,epoch,checkpoint_num)
            torch.save({
                #'current_class': single,
                'checkpoint_num': checkpoint_num,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'saved_train_loss': saved_train_losses,
                'saved_val_loss':saved_val_losses,
                'saved_x':saved_x,
                #'iters':iters,
                'train_acc':train_acc,
                'val_acc':val_acc
                }, './faster_model_saved/'+model_path+'.pth')

            checkpoint_num += 1
    # plotting
    plt.title("Training Curve")
    plt.plot(saved_x, saved_train_losses, label="Train")
    plt.plot(saved_x, saved_val_losses, label='Validation')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

    plt.title("Training Curve")
    plt.plot(saved_x, train_acc, label="Train")
    plt.plot(saved_x, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    end_time = time.time()
    duration = end_time - start_time

    print("Final Training Accuracy: {}".format(saved_train_acc[-1]))
    print("Final Validation Accuracy: {}".format(saved_val_acc[-1]))
    print ("Trained Duration: {} seconds".format(duration))

def train_net_continue_mul_gpu(model, batch_size=4, num_epochs=100, learning_rate = 0.0006,weight_decay=0.0002,ep=0,ck=0):

    model_path = "bs{0}_lr{1}_epoch{2}_checkpoint_{3}".format(batch_size,learning_rate,ep,ck)
    checkpoint = torch.load('./faster_model_saved/'+model_path+'.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    checkpoint_num = checkpoint['checkpoint_num']
    model = model.to(device)
    model = model.train()
    start_time = time.time()

    torch.manual_seed(1000)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)


    saved_x, saved_train_losses, saved_val_losses, train_acc, val_acc = [], [], [], [], []

    # training


    saved_idx = 0
    iter_acc = []
    iter_loss = []
    for epoch in range(ep,num_epochs):
        #idx = 0
        #import one class
        total_list = os.listdir('./train_data/')
        while 1:
            if total_list==[]:
                break
            current_file = torch.load('./train_data/'+total_list[0])
            total_list = total_list[1:]
            for element in current_file:
                batch_img = []
                batch_box = []
                for j in element:
                    img = j[0].to(device)
                    j[1]['boxes'] = j[1]['boxes']*640

                    box_label = j[1]
                    #box_label['boxes'] = box_label['boxes']*640

                    box_cuda = {
                      "boxes" : box_label['boxes'].to(device),
                      "labels" : box_label['labels'].to(device)
                    }
                    batch_img.append(img)
                    batch_box.append(box_cuda)

                loss_dict = model(batch_img,batch_box)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                iter_acc.append(get_train_accuracy_one_gpu(model,batch_box,batch_img))
                iter_loss.append(float(losses)/batch_size)
                print (('{} processing iterration {}... Train Loss: {} | Train Acc: {}').format(current_file[:-3],i,iter_loss[-1],iter_acc[-1]))

            saved_x.append(saved_idx)
            saved_idx += 1

            saved_train_losses.append(sum(iter_loss)/len(iter_loss))
            [current_val_loss,current_val_acc] = get_val_loss_acc_one_gpu(model.state_dict())
            saved_val_losses.append(current_val_loss)
            print ('-------------------------------------------------------------------------------------------')
            #print(('Train loss: {} | Validation loss: {}').format(saved_train_losses[-1],saved_val_losses[-1]))
            train_acc.append(sum(iter_acc)/len(iter_acc)) # compute training accuracy
            iter_acc = []
            iter_loss = []
            val_acc.append(current_val_acc)  # compute validation accuracy

            print(("Epoch {}: Train acc: {} |"+"Validation acc: {}").format(epoch + 1,train_acc[-1],val_acc[-1]))
            #print(("Epoch {}: Train acc: {} ").format(epoch + 1,train_acc[-1]))
            print ('-------------------------------------------------------------------------------------------')
            model_path = "bs{0}_lr{1}_epoch{2}_checkpoint_{3}".format(batch_size,learning_rate,epoch,checkpoint_num)
            torch.save({
                #'current_class': single,
                'checkpoint_num': checkpoint_num,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'saved_train_loss': saved_train_losses,
                'saved_val_loss':saved_val_losses,
                'saved_x':saved_x,
                #'iters':iters,
                'train_acc':train_acc,
                'val_acc':val_acc
                }, './faster_model_saved/'+model_path+'.pth')

            checkpoint_num += 1
    # plotting
    plt.title("Training Curve")
    plt.plot(saved_x, saved_train_losses, label="Train")
    plt.plot(saved_x, saved_val_losses, label='Validation')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

    plt.title("Training Curve")
    plt.plot(saved_x, train_acc, label="Train")
    plt.plot(saved_x, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    end_time = time.time()
    duration = end_time - start_time

    print("Final Training Accuracy: {}".format(saved_train_acc[-1]))
    print("Final Validation Accuracy: {}".format(saved_val_acc[-1]))
    print ("Trained Duration: {} seconds".format(duration))



def get_train_accuracy_one_gpu(model,box_labels,img):
    model = model.cuda(0)
    model = model.eval()
    out = model(img)
    total = 0
    correct = 0
    with torch.no_grad():
        for i in range(len(out)):
            if out[i]['labels'].tolist()!=[]:
                find_box = 0
                #for each in range(out[i]['labels'].shape[0]):
                #    if out[i]['labels'].tolist()[each]==box_labels[i]['labels'].tolist()[0]:
                #        correct += 1
                #        find_box = each
                #        break

                if out[i]['labels'].tolist()[find_box]==box_labels[i]['labels'].tolist()[0]:
                    correct += 1

                resize_box = np.array(out[i]['boxes'][find_box].cuda(0).tolist())#*0.375
                print(('\t\tpredicted {},{} |  expected {}, {}').format(out[i]['labels'][find_box].cuda(0),resize_box,box_labels[i]['labels'][0],box_labels[i]['boxes'][0].tolist()))

               # print(('\tpredicted {} | expected {}').format(out[i]['labels'][0].cuda(0),box_labels[i]['labels'][0]))
               # correct += out[i]['labels'].tolist()[0]==box_labels[i]['labels'].tolist()[0]
            else:
                print(('\t\tpredicted none |  expected {}, {}').format(box_labels[i]['labels'][0],box_labels[i]['boxes'][0].tolist()))
  
            #print('\tpredicted none')
            total += 1
    model = model.train()
    return correct/total

def get_val_loss_acc_two_gpu(parameters,batch_size=1):
    val_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=32).cuda(1)
    val_model.load_state_dict(parameters)
    val_model = val_model.train()

    s_losses = []
    val_idx = 0

    correct = 0
    total = 0
    with torch.no_grad():
        for single in classes:
            imported = torch.load('./tensor/validation/'+single+'.pt')
            for info in imported:
                img = info[0].cuda(1)
                info[1]['boxes'] = info[1]['boxes']*640

                box_label = info[1]
                #box_label['boxes'] = box_label['boxes']*640

                box_cuda = {
                        "boxes" : box_label['boxes'].cuda(1),
                        "labels" : box_label['labels'].cuda(1)
                }
                loss_dict = val_model([img.cuda(1)],[box_cuda])
                losses = sum(loss for loss in loss_dict.values())
                s_losses.append(losses/batch_size)
                del loss_dict
                val_model = val_model.eval()
                if val_model([img.cuda(1)])[0]['labels'].tolist()!=[]:
                    out = val_model([img.cuda(1)])[0]['labels'][0]
                    print(('\t\tpredicted {} and expected {}').format(out,box_cuda['labels'][0]))
                    if out==box_cuda['labels'][0]:
                        correct += 1
                    del out
                else:
                    print ('\t\tpredicted none')
                total += 1
                print (('\tprocessing iterration {}... Val Loss: {} | Val Acc: {}').format(val_idx,s_losses[-1],correct/total))
                val_idx += 1


                val_model = val_model.train()

    return [sum(s_losses)/len(s_losses),correct / total]

def train_net_two_gpu(model, batch_size=4, num_epochs=100, learning_rate = 0.0006,weight_decay=0.0002):
    checkpoint_num = 0
    model = model.cuda(0)
    model = model.train()
    start_time = time.time()

    torch.manual_seed(1000)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)


    saved_x, saved_train_losses, saved_val_losses, train_acc, val_acc = [], [], [], [], []

    # training


    saved_idx = 0
    iter_acc = []
    iter_loss = []
    for epoch in range(num_epochs):
        #idx = 0
        #import one class
        total_list = os.listdir('./train_data/')
        while 1:
            if total_list==[]:
                break
            current_file = torch.load('./train_data/'+total_list[0])
            total_list = total_list[1:]
            i = 0
            for element in current_file:
                batch_img = []
                batch_box = []
                for j in element:
                    img = j[0].cuda(0)
                    j[1]['boxes'] = j[1]['boxes']*640

                    box_label = j[1]
                    #box_label['boxes'] = box_label['boxes']*640

                    box_cuda = {
                      "boxes" : box_label['boxes'].cuda(0),
                      "labels" : box_label['labels'].cuda(0)
                    }
                    batch_img.append(img)
                    batch_box.append(box_cuda)

                loss_dict = model(batch_img,batch_box)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                iter_acc.append(get_train_accuracy_one_gpu(model,batch_box,batch_img))
                iter_loss.append(float(losses)/batch_size)
                i += 1
                print (('processing iterration {}... Train Loss: {} | Train Acc: {}').format(i,iter_loss[-1],iter_acc[-1]))

            saved_x.append(saved_idx)
            saved_idx += 1

            saved_train_losses.append(sum(iter_loss)/len(iter_loss))
            [current_val_loss,current_val_acc] = get_val_loss_acc_two_gpu(model.state_dict())
            saved_val_losses.append(current_val_loss)
            print ('-------------------------------------------------------------------------------------------')
            #print(('Train loss: {} | Validation loss: {}').format(saved_train_losses[-1],saved_val_losses[-1]))
            train_acc.append(sum(iter_acc)/len(iter_acc)) # compute training accuracy
            iter_acc = []
            iter_loss = []
            val_acc.append(current_val_acc)  # compute validation accuracy

            print(("Epoch {}: Train acc: {} |"+"Validation acc: {}").format(epoch + 1,train_acc[-1],val_acc[-1]))
            #print(("Epoch {}: Train acc: {} ").format(epoch + 1,train_acc[-1]))
            print ('-------------------------------------------------------------------------------------------')
            model_path = "bs{0}_lr{1}_epoch{2}_checkpoint_{3}".format(batch_size,learning_rate,epoch,checkpoint_num)
            torch.save({
                #'current_class': single,
                'checkpoint_num': checkpoint_num,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'saved_train_loss': saved_train_losses,
                'saved_val_loss':saved_val_losses,
                'saved_x':saved_x,
                #'iters':iters,
                'train_acc':train_acc,
                'val_acc':val_acc
                }, './faster_model_saved/'+model_path+'.pth')

            checkpoint_num += 1
    # plotting
    plt.title("Training Curve")
    plt.plot(saved_x, saved_train_losses, label="Train")
    plt.plot(saved_x, saved_val_losses, label='Validation')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

    plt.title("Training Curve")
    plt.plot(saved_x, train_acc, label="Train")
    plt.plot(saved_x, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    end_time = time.time()
    duration = end_time - start_time

    print("Final Training Accuracy: {}".format(saved_train_acc[-1]))
    print("Final Validation Accuracy: {}".format(saved_val_acc[-1]))
    print ("Trained Duration: {} seconds".format(duration))

def train_net_continue_two_gpu(model, batch_size=1, num_epochs=100, learning_rate = 0.0006,weight_decay=0.0002,ep=0,ck=0):

    model_path = "bs{0}_lr{1}_epoch{2}_checkpoint_{3}".format(batch_size,learning_rate,ep,ck)
    checkpoint = torch.load('./faster_model_saved/'+model_path+'.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    checkpoint_num = checkpoint['checkpoint_num']
    model = model.cuda(0)
    model = model.train()
    start_time = time.time()

    torch.manual_seed(1000)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)


    saved_x, saved_train_losses, saved_val_losses, train_acc, val_acc = [], [], [], [], []

    # training


    saved_idx = 0
    iter_acc = []
    iter_loss = []
    for epoch in range(ep,num_epochs):
        #idx = 0
        #import one class
        total_list = os.listdir('./train_data/')
        while 1:
            if total_list==[]:
                break
            current_file = torch.load('./train_data/'+total_list[0])
            total_list = total_list[1:]
            i = 0
            for element in current_file:
                batch_img = []
                batch_box = []
                for j in element:
                    img = j[0].cuda(0)
                    j[1]['boxes'] = j[1]['boxes']*640

                    box_label = j[1]
                    #box_label['boxes'] = box_label['boxes']*640

                    box_cuda = {
                      "boxes" : box_label['boxes'].cuda(0),
                      "labels" : box_label['labels'].cuda(0)
                    }
                    batch_img.append(img)
                    batch_box.append(box_cuda)

                loss_dict = model(batch_img,batch_box)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                iter_acc.append(get_train_accuracy_one_gpu(model,batch_box,batch_img))
                iter_loss.append(float(losses)/batch_size)
                i += 1
                print (('processing iterration {}... Train Loss: {} | Train Acc: {}').format(i,iter_loss[-1],iter_acc[-1]))

            saved_x.append(saved_idx)
            saved_idx += 1

            saved_train_losses.append(sum(iter_loss)/len(iter_loss))
            [current_val_loss,current_val_acc] = get_val_loss_acc_two_gpu(model.state_dict())
            saved_val_losses.append(current_val_loss)
            print ('-------------------------------------------------------------------------------------------')
            #print(('Train loss: {} | Validation loss: {}').format(saved_train_losses[-1],saved_val_losses[-1]))
            train_acc.append(sum(iter_acc)/len(iter_acc)) # compute training accuracy
            iter_acc = []
            iter_loss = []
            val_acc.append(current_val_acc)  # compute validation accuracy

            print(("Epoch {}: Train acc: {} |"+"Validation acc: {}").format(epoch + 1,train_acc[-1],val_acc[-1]))
            #print(("Epoch {}: Train acc: {} ").format(epoch + 1,train_acc[-1]))
            print ('-------------------------------------------------------------------------------------------')
            model_path = "bs{0}_lr{1}_epoch{2}_checkpoint_{3}".format(batch_size,learning_rate,epoch,checkpoint_num)
            torch.save({
                #'current_class': single,
                'checkpoint_num': checkpoint_num,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'saved_train_loss': saved_train_losses,
                'saved_val_loss':saved_val_losses,
                'saved_x':saved_x,
                #'iters':iters,
                'train_acc':train_acc,
                'val_acc':val_acc
                }, './faster_model_saved/'+model_path+'.pth')

            checkpoint_num += 1
    # plotting
    plt.title("Training Curve")
    plt.plot(saved_x, saved_train_losses, label="Train")
    plt.plot(saved_x, saved_val_losses, label='Validation')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

    plt.title("Training Curve")
    plt.plot(saved_x, train_acc, label="Train")
    plt.plot(saved_x, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    end_time = time.time()
    duration = end_time - start_time

    print("Final Training Accuracy: {}".format(saved_train_acc[-1]))
    print("Final Validation Accuracy: {}".format(saved_val_acc[-1]))
    print ("Trained Duration: {} seconds".format(duration))

def train_net_one_gpu(model, batch_size=4, num_epochs=100, learning_rate = 0.0006,weight_decay=0.0002,lr_decay=4):
    checkpoint_num = 0
    model = model.cuda(0)
    model = model.train()
    start_time = time.time()

    torch.manual_seed(1000)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
#    optimizer = optim.Adagrad(params,lr=learning_rate,weight_decay=weight_decay,lr_decay=lr_decay)
   

    saved_x, saved_train_losses, saved_val_losses, train_acc, val_acc = [], [], [], [], []

    # training


    saved_idx = 0
    iter_acc = []
    iter_loss = []
    for epoch in range(num_epochs):
        #idx = 0
        #import one class
        total_list = os.listdir('./train_data/')
        total_file = len(total_list)
        file_idx = 0
        while 1:
            if total_list==[]:
                break
            f_name = total_list[0]
            current_file = torch.load('./train_data/'+total_list[0])
            total_list = total_list[1:]
            i = 0
            for element in current_file:
                batch_img = []
                batch_box = []
                batch_box_cp = []
                for j in element:
                    img = j[0].cuda(0)
                    x = j[1].copy()
                    x['boxes'] = x['boxes']*640
                    #box_label = j[1]
                    #box_label['boxes'] = box_label['boxes']*640
                    #print ('box label')
                    #print (box_label['boxes'])
                    box_cuda = {
                      "boxes" : x['boxes'].cuda(0),
                      "labels" : x['labels'].cuda(0)
                    }
                    box_cp = box_cuda.copy()
                    #print ('box cuda')
                    #print (box_cuda['boxes'])
                    batch_img.append(img)
                    batch_box.append(box_cuda)
                    batch_box_cp.append(box_cp)
                #for i in batch_box:
                #    print (i['boxes'])


                loss_dict = model(batch_img,batch_box)
                losses = sum(loss for loss in loss_dict.values())
               # for i in batch_box:
               #     print (i['boxes'])

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                #for i in batch_box_cp:
                #    print (i['boxes'])
                #iter_acc.append(get_train_accuracy_one_gpu(model,batch_box_cp,batch_img))
                iter_loss.append(float(losses)/batch_size)
                i += 1
                print (('{}/{} processing iterration {}... Train Loss: {}').format(file_idx,total_file,i,iter_loss[-1]))
            file_idx += 1
            if file_idx%6==0:
                saved_x.append(saved_idx)
                saved_idx += 1
                iter_acc.append(get_train_accuracy_one_gpu_batch(model))
                saved_train_losses.append(sum(iter_loss)/len(iter_loss))
                [current_val_loss,current_val_acc] = get_val_loss_acc_one_gpu(model)
                saved_val_losses.append(current_val_loss)
                print ('-------------------------------------------------------------------------------------------')
                #print(('Train loss: {} | Validation loss: {}').format(saved_train_losses[-1],saved_val_losses[-1]))
                train_acc.append(sum(iter_acc)/len(iter_acc)) # compute training accuracy
                iter_acc = []
                iter_loss = []
                val_acc.append(current_val_acc)  # compute validation accuracy

                print(("Epoch {}: Train acc: {} |"+"Validation acc: {}").format(epoch + 1,train_acc[-1],val_acc[-1]))
                #print(("Epoch {}: Train acc: {} ").format(epoch + 1,train_acc[-1]))
                print ('-------------------------------------------------------------------------------------------')
                model_path = "bs{0}_lr{1}_epoch{2}_checkpoint_{3}".format(batch_size,learning_rate,epoch,checkpoint_num)
                torch.save({
                    #'current_class': single,
                    'checkpoint_num': checkpoint_num,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'saved_train_loss': saved_train_losses,
                    'saved_val_loss':saved_val_losses,
                    'saved_x':saved_x,
                    #'iters':iters,
                    'train_acc':train_acc,
                    'val_acc':val_acc
                    }, './faster_model_saved/'+model_path+'.pth')

                checkpoint_num += 1
    # plotting
    plt.title("Training Curve")
    plt.plot(saved_x, saved_train_losses, label="Train")
    plt.plot(saved_x, saved_val_losses, label='Validation')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

    plt.title("Training Curve")
    plt.plot(saved_x, train_acc, label="Train")
    plt.plot(saved_x, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    end_time = time.time()
    duration = end_time - start_time

    print("Final Training Accuracy: {}".format(saved_train_acc[-1]))
    print("Final Validation Accuracy: {}".format(saved_val_acc[-1]))
    print ("Trained Duration: {} seconds".format(duration))

def train_net_continue_one_gpu(model, batch_size=1, num_epochs=100, learning_rate = 0.0006,weight_decay=0.0002,lr_decay=4,ep=0,ck=0):

    model_path = "bs{0}_lr{1}_epoch{2}_checkpoint_{3}".format(batch_size,0.00005,ep,ck)
    checkpoint = torch.load('./faster_model_saved/'+model_path+'.pth')
    torch.manual_seed(1000)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
#    optimizer = optim.Adagrad(params,lr=learning_rate,weight_decay=weight_decay,lr_decay=lr_decay)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    checkpoint_num = checkpoint['checkpoint_num']
    model = model.cuda(0)
    model = model.train()
    start_time = time.time()

   
            

    saved_x = checkpoint['saved_x']
    saved_train_losses = checkpoint['saved_train_loss']
    saved_val_losses = checkpoint['saved_val_loss']
    train_acc = checkpoint['train_acc']
    val_acc = checkpoint['val_acc']
                                    
    saved_idx = 0                                                                        
    if saved_x!=[]:
        saved_idx = saved_x[-1]+1 
   # saved_idx = saved_x[-1]+1
    #saved_x, saved_train_losses, saved_val_losses, train_acc, val_acc = [], [], [], [], []

    iter_acc = []
    iter_loss = []
    for epoch in range(ep+1,num_epochs):
        #idx = 0
        #import one class
        total_list = os.listdir('./train_data/')
        total_file = len(total_list)
        file_idx = 0
        while 1:
            if total_list==[]:
                break
            f_name = total_list[0]
            current_file = torch.load('./train_data/'+total_list[0])
            total_list = total_list[1:]
            i = 0
            for element in current_file:
                batch_img = []
                batch_box = []
                batch_box_cp = []
                for j in element:
                    img = j[0].cuda(0)
                    j[1]['boxes'] = j[1]['boxes']*640

                    box_label = j[1]
                    #box_label['boxes'] = box_label['boxes']*640
                    box_cuda = {
                      "boxes" : box_label['boxes'].cuda(0),
                      "labels" : box_label['labels'].cuda(0)
                    }
                    box_cp = box_cuda.copy()
                    batch_img.append(img)
                    batch_box.append(box_cuda)
                    batch_box_cp.append(box_cp)

                loss_dict = model(batch_img,batch_box)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
               # for i in batch_box:
               #     print(i['boxes'])
                #iter_acc.append(get_train_accuracy_one_gpu(model,batch_box_cp,batch_img))
                iter_loss.append(float(losses)/batch_size)
                i += 1
                print (('{}/{} processing iterration {}... Train Loss: {}').format(file_idx,total_file,i,iter_loss[-1]))
            file_idx += 1
            if file_idx%6==0:
                iter_acc.append(get_train_accuracy_one_gpu_batch(model))

                saved_x.append(saved_idx)
                saved_idx += 1

                saved_train_losses.append(sum(iter_loss)/len(iter_loss))
                [current_val_loss,current_val_acc] = get_val_loss_acc_one_gpu(model)
                saved_val_losses.append(current_val_loss)
                print ('-------------------------------------------------------------------------------------------')
                print(('Train loss: {} | Validation loss: {}').format(saved_train_losses[-1],saved_val_losses[-1]))
                train_acc.append(sum(iter_acc)/len(iter_acc)) # compute training accuracy
                iter_acc = []
                iter_loss = []
                val_acc.append(current_val_acc)  # compute validation accuracy

                print(("Epoch {}: Train acc: {} |"+"Validation acc: {}").format(epoch + 1,train_acc[-1],val_acc[-1]))
                #print(("Epoch {}: Train acc: {} ").format(epoch + 1,train_acc[-1]))
                print ('-------------------------------------------------------------------------------------------')
                checkpoint_num += 1   
                model_path = "bs{0}_lr{1}_epoch{2}_checkpoint_{3}".format(batch_size,learning_rate,epoch,checkpoint_num)
                torch.save({
                    #'current_class': single,
                    'checkpoint_num': checkpoint_num,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'saved_train_loss': saved_train_losses,
                    'saved_val_loss':saved_val_losses,
                    'saved_x':saved_x,
                    #'iters':iters,
                    'train_acc':train_acc,
                    'val_acc':val_acc
                    }, './faster_model_saved/'+model_path+'.pth')

    
    # plotting
    plt.title("Training Curve")
    plt.plot(saved_x, saved_train_losses, label="Train")
    plt.plot(saved_x, saved_val_losses, label='Validation')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

    plt.title("Training Curve")
    plt.plot(saved_x, train_acc, label="Train")
    plt.plot(saved_x, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    end_time = time.time()
    duration = end_time - start_time

    print("Final Training Accuracy: {}".format(saved_train_acc[-1]))
    print("Final Validation Accuracy: {}".format(saved_val_acc[-1]))
    print ("Trained Duration: {} seconds".format(duration))


def get_val_loss_acc_one_gpu(val_model,batch_size=1):
    s_losses = []
    val_idx = 0
    correct = 0
    total = 0
    with torch.no_grad():
        val_model = val_model.train()
        for single in classes:
            imported = torch.load('./tensor/validation/'+single+'.pt')
            index = 0
            for info in imported:
                if index>=80:
                    break
                index += 1
                img = info[0].cuda(0)
                info[1]['boxes'] = info[1]['boxes']*640

                box_label = info[1]
                #box_label['boxes'] = box_label['boxes']*640

                box_cuda ={
                        "boxes" : box_label['boxes'].cuda(0),
                        "labels" : box_label['labels'].cuda(0)
                }
                box_cp = box_cuda.copy()
                loss_dict = val_model([img.cuda(0)],[box_cuda])
                losses = sum(loss for loss in loss_dict.values())
                s_losses.append(losses/batch_size)
                del loss_dict
                val_model = val_model.eval()
                if val_model([img.cuda(0)])[0]['labels'].tolist()!=[]:
                    predict =  val_model([img.cuda(0)])[0]
                    find_idx = 0
                    for each in range(predict['labels'].shape[0]):
                        if predict['labels'].tolist()[each]==box_cp['labels'].tolist()[0]:
                            correct += 1
                            find_idx = each
                            break
                    out = predict['labels'][find_idx]
                    resize_box = np.array(predict['boxes'][find_idx].tolist())#*0.375 
                    print(('\t\tpredicted {},{} |  expected {}, {}').format(out,resize_box,box_cp['labels'][0],box_cp['boxes'][0].tolist()))
                    #if predict['labels'].tolist()[0]==box_cp['labels'].tolist()[0]:
                    #    correct += 1
                    #del out
                else:
                    print(('\t\tpredicted none |  expected {}, {}').format(box_cp['labels'][0],box_cp['boxes'][0].tolist()))

                total += 1
                print (('\tprocessing iterration {}... Val Loss: {} | Val Acc: {}').format(val_idx,s_losses[-1],correct/total))
                val_idx += 1
                val_model = val_model.train()

    return [sum(s_losses)/len(s_losses),correct / total]

def get_train_accuracy_one_gpu_batch(model):
    model = model.cuda(0)
    model = model.eval()
    #out = model(img)
    total = 0
    correct = 0
    num_iter = 0
    with torch.no_grad():
        total_list = os.listdir('./train_data/')
        total_file = len(total_list)
        file_idx = 0
        while 1:
            if total_list==[]:
                break
            f_name = total_list[0]
            current_file = torch.load('./train_data/'+total_list[0])
            total_list = total_list[1:]
            #i = 0
            for element in current_file:
                batch_img = []
                batch_box = []
                #batch_box_cp = []
                for j in element:
                    img = j[0].cuda(0)
                    j[1]['boxes'] = j[1]['boxes']*640

                    box_label = j[1]
                    #box_label['boxes'] = box_label['boxes']*640
                    box_cuda = {
                      "boxes" : box_label['boxes'].cuda(0),
                      "labels" : box_label['labels'].cuda(0)
                    }
                   # box_cp = box_cuda.copy()
                    batch_img.append(img)
                    batch_box.append(box_cuda)
                   # batch_box_cp.append(box_cp)

                out = model(batch_img)

                for i in range(len(out)):
                    total += 1
                    if out[i]['labels'].tolist()!=[]:
                        find_box = 0
#                        for each in range(out[i]['labels'].shape[0]):
#                            if out[i]['labels'].tolist()[each]==batch_box[i]['labels'].tolist()[0]:
#                                correct += 1
#                                find_box = each
#                                break
#
                        if out[i]['labels'].tolist()[find_box]==batch_box[i]['labels'].tolist()[0]:
                            correct += 1

                        resize_box = np.array(out[i]['boxes'][find_box].cuda(0).tolist())#*0.375
                        print(('predicted {},{} |  expected {}, {}').format(out[i]['labels'][find_box].cuda(0),resize_box,batch_box[i]['labels'][0],batch_box[i]['boxes'][0].tolist()))

               # print(('\tpredicted {} | expected {}').format(out[i]['labels'][0].cuda(0),box_labels[i]['labels'][0]))
               # correct += out[i]['labels'].tolist()[0]==box_labels[i]['labels'].tolist()[0]
                    else:
                        print(('predicted none |  expected {}, {}').format(batch_box[i]['labels'][0],batch_box[i]['boxes'][0].tolist()))
                num_iter += 1
                print (('\t{}/{} train accuracy: {}').format(file_idx,total_file,correct/total)) 
            file_idx += 1
            #print('\tpredicted none')
            #total += 1
    model = model.train()
    return correct/total


#multiple

#rcnn_v1 = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=32).to(device)
#rcnn_v1 = nn.DataParallel(rcnn_v1)
#train_net_mul_gpu(rcnn_v1, batch_size=8, num_epochs=50, learning_rate = 0.0001,weight_decay=0.0001)


#single
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 200),),aspect_ratios=((0.5, 1.0, 2.0),))
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],output_size=7,sampling_ratio=2)
rcnn_v1 = FasterRCNN(backbone,num_classes=32,rpn_anchor_generator=anchor_generator,box_roi_pool=roi_pooler).cuda(0)

#rcnn_v1 = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=32).cuda(0)
#train_net_one_gpu(rcnn_v1, batch_size=8, num_epochs=50, learning_rate = 0.0001,weight_decay=0.0001)
train_net_continue_one_gpu(rcnn_v1, batch_size=8, num_epochs=50, learning_rate = 0.00003,weight_decay=0.0001,ep=2,ck=15)
