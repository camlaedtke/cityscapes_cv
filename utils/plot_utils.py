import numpy as np
import matplotlib.pyplot as plt


def plot_history(results, u2net=False):
    
    plt.figure(figsize=(15,7))

    plt.subplot(1,3,1)  
    if u2net:
        plt.plot(results.history['d0_loss'], 'r', label='Training loss')
        plt.plot(results.history['val_d0_loss'], 'b', label='Validation loss')
    else: 
        plt.plot(results.history['loss'], 'r', label='Training loss')
        plt.plot(results.history['val_loss'], 'b', label='Validation loss')
    plt.title('Log Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.legend(prop={'size': 14})

    plt.subplot(1,3,2)
    if u2net:
        plt.plot(results.history['d0_accuracy'], 'r', label='Training accuracy')
        plt.plot(results.history['val_d0_accuracy'], 'b', label='Validation accuracy')
    else:
        plt.plot(results.history['accuracy'], 'r', label='Training accuracy')
        plt.plot(results.history['val_accuracy'], 'b', label='Validation accuracy')
    plt.title('Accuracy', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.legend(prop={'size': 14})

    plt.subplot(1,3,3)
    if u2net:
        plt.plot(results.history['d0_dice_coef'], 'r', label='Dice coefficient')
        plt.plot(results.history['val_d0_dice_coef'], 'b', label='Validation dice coefficient')
    else:
        plt.plot(results.history['dice_coef'], 'r', label='Dice coefficient')
        plt.plot(results.history['val_dice_coef'], 'b', label='Validation dice coefficient')
    plt.title('Dice Coefficient', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.legend(prop={'size': 14})
    plt.show()
    
    
    
def plot_dice_and_iou(trainId2label, n_classes, class_dice, class_iou):
    categories = [trainId2label[i].category for i in range(n_classes)]
    cmap = [color['color'] for color in plt.rcParams['axes.prop_cycle']]
    cat_colors = {
        'void': 'black',
        'flat': cmap[0],
        'construction': cmap[1],
        'object': cmap[2],
        'nature': cmap[3],
        'sky': cmap[4],
        'human': cmap[5],
        'vehicle': cmap[6]
    }
    colors = [cat_colors[category] for category in categories]

    names = [trainId2label[i].name for i in range(n_classes)]

    plt.style.use('ggplot')


    plt.figure(figsize=(15,20))

    plt.subplot(2,1,1)
    plt.barh(names, class_dice, color=colors)
    plt.xlabel("Dice Coefficient", fontsize=18)
    plt.ylabel("Class Name", fontsize=18)
    plt.title("Class Dice Scores", fontsize=22)
    plt.xlim([0, 1])

    plt.subplot(2,1,2)
    plt.barh(names, class_iou, color=colors)
    plt.xlabel("Intersection Over Union", fontsize=18)
    plt.ylabel("Class Name", fontsize=18)
    plt.title("Class IOU Scores", fontsize=22)
    plt.xlim([0, 1])
    plt.show()
    
   
    
def plot_dice(trainId2label, n_classes, class_dice):
    categories = [trainId2label[i].category for i in range(n_classes)]
    cmap = [color['color'] for color in plt.rcParams['axes.prop_cycle']]
    cat_colors = {
        'void': 'black',
        'flat': cmap[0],
        'construction': cmap[1],
        'object': cmap[2],
        'nature': cmap[3],
        'sky': cmap[4],
        'human': cmap[5],
        'vehicle': cmap[6]
    }
    colors = [cat_colors[category] for category in categories]

    names = [trainId2label[i].name for i in range(n_classes)]

    plt.figure(figsize=(14,10), dpi=200)
    plt.barh(names, class_dice, color=colors)
    plt.xlabel("Dice Coefficient", fontsize=18)
    plt.ylabel("Class Name", fontsize=18)
    plt.title("Class Dice Scores", fontsize=22)
    plt.xlim([0, 1])
    plt.show()