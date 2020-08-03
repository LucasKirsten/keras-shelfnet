import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K

class Model():
    
    def __init__(self,
              model_name,
              classes,
              input_shape   = (224,224,3),
              dropout       = 0.2,
              learning_rate = 1e-3,
              loss          = 'binary_crossentropy',
              metrics       = [jaccard_distance, 'acc']):
        
        model_name    = self.model_name
        classes       = self.classes
        input_shape   = self.input_shape
        dropout       = self.dropout
        learning_rate = self.learning_rate
        loss          = self.loss
        metrics       = self.metrics
    
    def build(self):
        
        def ConvBloc(input_tensor, n_filters, ksize, strides=1):
            ''' ResNet layer block '''
            u = L.SeparableConv2D(filters=n_filters, kernel_size=ksize, padding='same') (input_tensor)
            u = L.BatchNormalization() (u)
            u = L.Activation('relu') (u)
            
            u = L.SeparableConv2D(filters=n_filters, kernel_size=ksize, padding='same') (u)
            u = L.BatchNormalization() (u)
            
            c = L.Conv2D(n_filters, 1, padding='same') (input_tensor)
            
            u = L.add([u, c])
            u = L.Activation('relu') (u)
            return u
        
        def DimReduceBlock(input_tensor, n_filters, ksize, dropout):
            p = ConvBloc(input_tensor, n_filters, ksize)
            p = L.MaxPooling2D((2, 2)) (p)
            p = L.Dropout(dropout) (p)
            return p
        
        def ConvTrans(input_tensor, n_filters):
            ''' Modified Conv2DTranspose using UpSampling2D and SeparableConv2D '''
            u = L.UpSampling2D() (input_tensor)
            u = L.SeparableConv2D(filters=n_filters, kernel_size=(2,2), padding='same') (u)
            return u
        
        def LateralConv(input_tensor, n_filters):
            ''' Lateral Convolution to S-Block inputs '''
            u = L.Conv2D(n_filters, 1, padding='same') (input_tensor)
            u = L.BatchNormalization() (u)
            u = L.Activation('relu') (u)
            return u
        
        def SBlock(lateral_input, vertical_input, n_filters, ksize, dropout):
            Shared = L.SeparableConv2D(filters=n_filters, kernel_size=ksize, padding='same') # shared weigth convolution
            
            f = L.add([lateral_input, vertical_input])
            u = Shared(f)
            u = L.BatchNormalization() (u)
            u = L.Activation('relu') (u)
            u = L.Dropout(dropout) (u)
            u = Shared(u)
            u = L.BatchNormalization() (u)
            
            u = L.add([f, u])
            u = L.Activation('relu') (u)
            return u
        
        filters = [(64,3),(128,3),(256,3),(512,3)]

        input_tensor = L.Input(self.input_shape)
        
        # first convolution
        p = L.Conv2D(filters[0][0], 5, use_bias=bias, padding='same') (input_tensor)

        # backbone
        backbone_layers = []
        for (f,k) in filters:
            p = DimReduceBlock(p, f, k, self.dropout)
            # (1)
            l = LateralConv(p, f)
            backbone_layers.append(l)

        # (2) S-Block decoder
        lateral_outputs_2 = []
        for bl, (f,k) in zip(backbone_layers[::-1], filters[::-1]):
            p = SBlock(bl, p, f, k, self.dropout)
            p = ConvTrans(p, f//2)
            lateral_outputs_2.append(p)
            
        # (3) S-Block encoder
        lateral_outputs_3 = []
        for lo, (f,k) in zip(lateral_outputs_2[::-1], filters):
            p = SBlock(lo, p, f//2, k, self.dropout)
            p = L.SeparableConv2D(filters=f, kernel_size=k, strides=(2,2), padding='same') (p)
            lateral_outputs_3.append(p)
            
        # (4) S-Block decoder
        for lo, (f,k) in zip(lateral_outputs_3[::-1], filters[::-1]):
            p = SBlock(lo, p, f, k, self.dropout)
            p = ConvTrans(p, f//2)

        mask    = L.Conv2D(self.classes, (1, 1), activation='sigmoid', name='mask_output') (p)

        model =  tf.keras.models.Model(input_tensor, mask, name=self.model_name)
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                      loss=self.loss,
                      metrics=self.metrics)
        
        return model