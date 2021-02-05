import numpy as np

training_data=open('dataset/mnist_train.csv','r')
training_data_list=training_data.readlines()
training_data.close()

epochs=3

inputs_number=784
neurons_number=530 # можно "поиграться" с этим значением
outputs_number=10

learning_rate=0.15 # скорость обучения,  можно "поиграться" с этим значением


fst_layer_weights = np.random.normal(0.0, pow(inputs_number, -0.5), (neurons_number, inputs_number))
snd_layer_weights = np.random.normal(0.0, pow(neurons_number, -0.5), (outputs_number, neurons_number))




# fst_layer_weights = (np.random.rand(neurons_number, inputs_number) -0.5)
# snd_layer_weights = (np.random.rand(outputs_number, neurons_number) -0.5)



def training(inputs_list,targets_list,fst_w,snd_w):
    inputs=np.array(inputs_list, ndmin=2).T
    targets=np.array(targets_list, ndmin=2).T

    x1=np.dot(fst_w, inputs)
    y1=1/(1+np.exp(-x1))
    
    x2=np.dot(snd_w, y1)
    y2=1/(1+np.exp(-x2))

    E =-(targets-y2)
    E_h =np.dot( snd_w.T, E)

    snd_w-=learning_rate*np.dot((E * y2 * (1.0-y2)), np.transpose(y1))
    fst_w-=learning_rate*np.dot((E_h * y1 * (1.0-y1)),np.transpose(inputs))

    



for i in range(epochs):
    for j in training_data_list:
        line=j.split(',')

        inputs_list=(np.asfarray(line[1:])/255*0.99)+0.01

        targets_list=np.zeros(outputs_number)+0.01
        targets_list[int(line[0])]=0.99

        training(inputs_list,targets_list,fst_layer_weights,snd_layer_weights)
            



print('Весовые коэффициенты:\n', fst_layer_weights)
print('Весовые коэффициенты:\n', snd_layer_weights)

test_data=open('dataset/mnist_test.csv','r')
test_data_list=test_data.readlines()
test_data.close()

efficiency=[]


def testing(inputs_list,fst_w,snd_w):
    inputs=np.array(inputs_list, ndmin=2).T

    x1=np.dot(fst_w, inputs)
    y1=1/(1+np.exp(-x1))
    
    x2=np.dot(snd_w, y1)
    y2=1/(1+np.exp(-x2))

    return y2

    


for j in test_data_list:
    line=j.split(',')

    inputs_list=(np.asfarray(line[1:])/255*0.99)+0.01

    target=int(line[0])

    outputs=testing(inputs_list,fst_layer_weights,snd_layer_weights)

    max_output_index=np.argmax(outputs)

    if (max_output_index == target):
        efficiency.append(1)
    else:
        efficiency.append(0)

efficiency_array=np.asarray(efficiency)

performance=(efficiency_array.sum() / efficiency_array.size)*100

print('Производительность:',performance,'%')



w1_mat = np.matrix(fst_layer_weights)
# with open('weights/fst_w.txt','wb') as f:
#     for line in w1_mat:
#         np.savetxt(f, line, fmt='%.8f')
np.savetxt("weights/fst_w.csv", w1_mat, delimiter=",")
print("Успешно сохранены веса из первого слоя.")

w2_mat = np.matrix(snd_layer_weights)
# with open('weights/snd_w.txt','wb') as f:
#     for line in w2_mat:
#         np.savetxt(f, line, fmt='%.8f')
np.savetxt("weights/snd_w.csv", w2_mat, delimiter=",")
print("Успешно сохранены веса из второго слоя.")




