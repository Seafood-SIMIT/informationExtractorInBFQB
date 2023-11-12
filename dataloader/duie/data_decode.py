
import numpy as np
def labelDecode(text,predict):


        predict_entity = predict['predict_entity'].cpu().detach().numpy()
        ans = np.argmax(predict_entity[0],axis=1)
        # shiti start
        print(text[0])
        text = text[0]
        print('object:',"".join([text[i] for i in np.where(ans==1)[0]]+
                            [text[i] for i in np.where(ans==2)[0]]+
                            [text[i] for i in np.where(ans==3)[0]]))
        print('subject:',"".join([text[i] for i in np.where(ans==4)[0]]+
                            [text[i] for i in np.where(ans==5)[0]]+
                            [text[i] for i in np.where(ans==6)[0]]))

        predict_anchor = predict['predict_anchor'].cpu().detach().numpy()
        predict_relation = predict['predict_relation'].cpu().detach().numpy()
        bilive = predict_anchor[0,:,0]
        start = predict_anchor[0,:,1]
        length = predict_anchor[0,:,2]
        for anchor_ids in range(10):
            print(bilive[anchor_ids],text[int(400*start[anchor_ids])],text[int(400*(start[anchor_ids]+length[anchor_ids]))])
        print('='*20)
    
        #print(text[i],'\t',shiti_s[i]>0.1)
    #shiti_o = shiti[]