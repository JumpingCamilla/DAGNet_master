import torch
import torch.nn as nn

from . import networks
from . import layers

from functools import partial
import pdb
import torch.nn.functional as F
class DADepthCommon(nn.Module):
    def __init__(self, num_layers, split_pos, grad_scales=(0.9, 0.1), pretrained=False):
        super().__init__()

        self.encoder = networks.ResnetEncoder(num_layers, pretrained)
        self.num_layers = num_layers  # This information is needed in the train loop for the sequential training

        # Number of channels for the skip connections and internal connections
        # of the decoder network, ordered from input to output
        self.shape_enc = tuple(reversed(self.encoder.num_ch_enc))
        self.shape_dec = (256, 128, 64, 32, 16)

        self.decoder = networks.PartialDecoder.gen_head(self.shape_dec, self.shape_enc, split_pos)
        self.split = layers.ScaledSplit(*grad_scales)

    def set_gradient_scales(self, depth, segmentation):
        self.split.set_scales(depth, segmentation)

    def get_gradient_scales(self):
        return self.split.get_scales()

    def forward(self, x):
        # The encoder produces outputs in the order
        # (highest res, second highest res, …, lowest res)
        x = self.encoder(x)

        # The decoder expects it's inputs in the order they are
        # used. E.g. (lowest res, second lowest res, …, highest res)
        x = tuple(reversed(x))

        # Replace some elements in the x tuple by decoded
        # tensors and leave others as-is
        x = self.decoder(*x) # CHANGE ME BACK TO THIS

        # Setup gradient scaling in the backward pass
        x = self.split(*x)

        # Experimental Idea: We want the decoder layer to be trained, so pass x to the decoder AFTER x was passed
        # to self.split which scales all gradients to 0 (if grad_scales are 0)
        # x = (self.decoder(*x[0]), ) + (self.decoder(*x[1]), ) + (self.decoder(*x[2]), )

        return x

    #def get_last_shared_layer(self):
    #    return self.encoder.encoder.layer4


class DADepthDepth(nn.Module):
    def __init__(self, common, resolutions=1):
        super().__init__()

        self.resolutions = resolutions

        self.decoder = networks.PartialDecoder.gen_tail(common.decoder)
        self.multires = networks.MultiResDepth(self.decoder.chs_x()[-resolutions:])

    def forward(self, *x):
        x = self.decoder(*x)
        x = self.multires(*x[-self.resolutions:])
        return x


class DADepthSeg(nn.Module):
    def __init__(self, common, resolutions=1, semantic_guidance=False):
        super().__init__()
        self.resolutions = resolutions
        self.semantic_guidance = semantic_guidance
        self.decoder = networks.PartialDecoder.gen_tail(common.decoder)
        if self.semantic_guidance:
            self.multires = networks.MultiResSegmentation(self.decoder.chs_x()[-resolutions:])
        else:
            self.multires = networks.MultiResSegmentation(self.decoder.chs_x()[-1:])
        self.nl = nn.Softmax2d()

    def forward(self, *x):
        x = self.decoder(*x)
        if self.semantic_guidance:
            x = self.multires(*x[-self.resolutions:])
            x_lin = x
        else:
            x = self.multires(*x[-1:])
            x_lin = x[-1]

        return x_lin


class DADepthPose(nn.Module):
    def __init__(self, num_layers, pretrained=False):
        super().__init__()

        self.encoder = networks.ResnetEncoder(
            num_layers, pretrained, num_input_images=2
        )

        self.decoder = networks.PoseDecoder(self.encoder.num_ch_enc[-1])

    def _transformation_from_axisangle(self, axisangle):
        n, h, w = axisangle.shape[:3]

        angles = axisangle.norm(dim=3)
        axes = axisangle / (angles.unsqueeze(-1) + 1e-7)

        # Implement the matrix from [1] with an additional identity fourth dimension
        # [1]: https://en.wikipedia.org/wiki/Transformation_matrix#Rotation_2

        angles_cos = angles.cos()
        angles_sin = angles.sin()

        res = torch.zeros(n, h, w, 4, 4, device=axisangle.device)
        res[:,:,:,:3,:3] = (1 - angles_cos.view(n,h,w,1,1)) * (axes.unsqueeze(-2) * axes.unsqueeze(-1))

        res[:,:,:,0,0] += angles_cos
        res[:,:,:,1,1] += angles_cos
        res[:,:,:,2,2] += angles_cos

        sl = axes[:,:,:,0] * angles_sin
        sm = axes[:,:,:,1] * angles_sin
        sn = axes[:,:,:,2] * angles_sin

        res[:,:,:,0,1] -= sn
        res[:,:,:,1,0] += sn

        res[:,:,:,1,2] -= sl
        res[:,:,:,2,1] += sl

        res[:,:,:,2,0] -= sm
        res[:,:,:,0,2] += sm

        res[:,:,:,3,3] = 1.0

        return res

    def _transformation_from_translation(self, translation):
        n, h, w = translation.shape[:3]

        # Implement the matrix from [1] with an additional dimension
        # [1]: https://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations

        res = torch.zeros(n, h, w, 4, 4, device=translation.device)
        res[:,:,:,:3,3] = translation
        res[:,:,:,0,0] = 1.0
        res[:,:,:,1,1] = 1.0
        res[:,:,:,2,2] = 1.0
        res[:,:,:,3,3] = 1.0

        return res

    def forward(self, x, invert):
        x = self.encoder(x)
        x = x[-1]  # take only the feature map of the last layer ...

        x_axisangle, x_translation = self.decoder(x)  # ... and pass it through the decoder

        x_rotation = self._transformation_from_axisangle(x_axisangle)

        if not invert:
            x_translation = self._transformation_from_translation(x_translation)

            return x_translation @ x_rotation

        else:
            x_rotation = x_rotation.transpose(3, 4)
            x_translation = -x_translation

            x_translation = self._transformation_from_translation(x_translation)

            return x_rotation @ x_translation


class DADepth(nn.Module):
    KEY_FRAME_CUR = ('color_aug', 0, 0)
    KEY_FRAME_PREV = ('color_aug', -1, 0)
    KEY_FRAME_NEXT = ('color_aug', 1, 0)

    def __init__(self, split_pos=1, num_layers=18, grad_scale_depth=0.95, grad_scale_seg=0.05,
                 weights_init='pretrained', resolutions_depth=1, num_layers_pose=18, semantic_guidance=False, depth_guidance=False, feature_fusion=False):

        super().__init__()

        # DAdepth allowed for five possible split positions.
        # The PartialDecoder developed as part of DAdepth
        # is a bit more flexible and allows splits to be
        # placed in between DAdepths splits.
        # As this class is meant to maximize compatibility
        # with DAdepth the line below translates between
        # the split position definitions.
        split_pos = max((2 * split_pos) - 1, 0)

        self.semantic_guidance = semantic_guidance
        self.depth_guidance = depth_guidance
        self.feature_fusion = feature_fusion
        # The Depth and the Segmentation Network have a common (=shared)
        # Encoder ("Feature Extractor")
        self.common = DADepthCommon(
            num_layers, split_pos, (grad_scale_depth, grad_scale_seg),
            weights_init == 'pretrained'
        )

        '''
        self.common = DADepthCommon(
            num_layers, split_pos, (grad_scale_depth, grad_scale_seg),
            weights_init == 'pretrained'
        )
        self.seg_common = DADepthCommon(
            num_layers, split_pos, (0, 1),
            weights_init == 'pretrained'
        )
        '''

        # While Depth and Seg Network have a shared Encoder,
        # each one has it's own Decoder
        self.depth = DADepthDepth(self.common, resolutions_depth)
        self.seg = DADepthSeg(self.common, resolutions_depth, semantic_guidance)
        #self.seg = DADepthSeg(self.seg_common, resolutions_depth, semantic_guidance)

        # The Pose network has it's own Encoder ("Feature Extractor") and Decoder
        self.pose = DADepthPose(
            num_layers_pose,
            weights_init == 'pretrained'
        )
        chs = tuple([20, 20, 20, 20])
        if self.semantic_guidance:
            self.multires = networks.MultiResGuidance(chs[-resolutions_depth:])

        if self.feature_fusion:
            self.attentionhead = networks.MHCA(256, num_heads=8)

            # norm_layer = partial(nn.LayerNorm, eps=1e-6)
            # self.attentionhead = networks.Block(dim=256, num_heads=8, mlp_hidden_dim=1024, qkv_bias=False, qk_scale=None,
            #     drop=0.1, attn_drop=0., drop_path=0, norm_layer=norm_layer)

    def _batch_pack(self, group):
        # Concatenate a list of tensors and remember how
        # to tear them apart again

        group = tuple(group)

        dims = tuple(b.shape[0] for b in group)  # dims = (DEFAULT_DEPTH_BATCH_SIZE, DEFAULT_SEG_BATCH_SIZE)
        group = torch.cat(group, dim=0)  # concatenate along the first axis, so along the batch axis

        return dims, group

    def _multi_batch_unpack(self, dims, *xs):
        xs = tuple(
            tuple(x.split(dims))
            for x in xs
        )

        # xs, as of now, is indexed like this:
        # xs[ENCODER_LAYER][DATASET_IDX], the lines below swap
        # this around to xs[DATASET_IDX][ENCODER_LAYER], so that
        # xs[DATASET_IDX] can be fed into the decoders.
        xs = tuple(zip(*xs))

        return xs

    def _check_purposes(self, dataset, purpose):
        # mytransforms.AddKeyValue is used in the loaders
        # to give each image a tuple of 'purposes'.
        # As of now these purposes can be 'depth' and 'segmentation'.
        # The torch DataLoader collates these per-image purposes
        # into list of them for each batch.
        # Check all purposes in this collated list for the requested
        # purpose (if you did not do anything wonky all purposes in a
        # batch should be equal),

        for purpose_field in dataset['purposes']:
            if purpose_field[0] == purpose:
                return True

    def set_gradient_scales(self, depth, segmentation):
        self.common.set_gradient_scales(depth, segmentation)

    def get_gradient_scales(self):
        return self.common.get_gradient_scales()

    def forward(self, batch):
        # Stitch together all current input frames
        # in the input group. So that batch normalization
        # in the encoder is done over all datasets/domains.
        dims, x = self._batch_pack(
            dataset[self.KEY_FRAME_CUR]
            for dataset in batch
        )

        # Feed the stitched-together input tensor through
        # the common network part and generate two output
        # tuples that look exactly the same in the forward
        # pass, but scale gradients differently in the backward pass.
        x_depth, x_seg = self.common(x)
        # x_depth, _ = self.common(x)
        # _, x_seg = self.seg_common(x)


        ## use the first and second scale of x_depth and x_seg to do multi head attention
        if self.feature_fusion:
            x_depth = list(x_depth)
            x_seg = list(x_seg)
            B, C, H, W = x_depth[0].shape
            #x_depth[0] = x_depth[0].reshape(B, C, H * W).permute(0, 2, 1)
            x_depth0 = x_depth[0].reshape(B, C, H * W).permute(0, 2, 1)
            x_seg[0] = x_seg[0].reshape(B, C, H * W).permute(0, 2, 1)
            #x_seg0 = x_seg[0].reshape(B, C, H * W).permute(0, 2, 1)
            #x_depth0 = self.attentionhead(x_seg0, x_depth[0], x_depth[0]).permute(0, 2, 1).reshape(B, C, H, W)
            #x_depth0 = self.attentionhead(x_depth[0], x_depth[0], x_depth[0]).permute(0, 2, 1).reshape(B, C, H, W)
            #x_seg0 = self.attentionhead(x_depth[0], x_seg[0], x_seg[0]).permute(0, 2, 1).reshape(B, C, H, W)
            x_seg0 = self.attentionhead(x_depth0, x_seg[0], x_seg[0]).permute(0, 2, 1).reshape(B, C, H, W)
            #x_depth[0] = x_depth0
            x_seg[0] = x_seg0
            B1, C1, H1, W1 = x_depth[1].shape
            #x_depth[1] = x_depth[1].reshape(B1, C1, H1 * W1).permute(0, 2, 1)
            x_depth1 = x_depth[1].reshape(B1, C1, H1 * W1).permute(0, 2, 1)
            x_seg[1] = x_seg[1].reshape(B1, C1, H1 * W1).permute(0, 2, 1)
            #x_seg1 = x_seg[1].reshape(B1, C1, H1 * W1).permute(0, 2, 1)
            #x_depth1 = self.attentionhead(x_depth[1], x_depth[1], x_depth[1]).permute(0, 2, 1).reshape(B1, C1, H1, W1)
            #x_depth1 = self.attentionhead(x_seg[1], x_depth[1], x_depth[1]).permute(0, 2, 1).reshape(B1, C1, H1, W1)
            #x_seg1 = self.attentionhead(x_depth[1], x_seg[1], x_seg[1]).permute(0, 2, 1).reshape(B1, C1, H1, W1)
            x_seg1 = self.attentionhead(x_depth1, x_seg[1], x_seg[1]).permute(0, 2, 1).reshape(B1, C1, H1, W1)
            #x_depth[1] = x_depth1
            x_seg[1] = x_seg1
            x_depth = tuple(x_depth)
            x_seg = tuple(x_seg)


        # Cut the stitched-together tensors along the
        # dataset boundaries so further processing can
        # be performed on a per-dataset basis.
        # x[DATASET_IDX][ENCODER_LAYER]
        x_depth = self._multi_batch_unpack(dims, *x_depth)
        x_seg = self._multi_batch_unpack(dims, *x_seg)

         ## Zhuy 22/05/29
        # if self.feature_fusion:
        #     x_depth = list(x_depth)
        #     x_seg = list(x_seg)
        #     try:
        #         x_depth[1] =list(x_depth[1])
        #         x_seg[1] = list(x_seg[1])
        #
        #         x_depth0 = x_depth[1][0]      #the first dimension: 0 images on kitti 1 images on Cityscape; the second dimension includes images from 4 scales
        #         B, C, H, W = x_depth0.shape
        #         x_depth0 = x_depth0.reshape(B, C, H * W).permute(0, 2, 1)
        #         x_seg[1][0] = x_seg[1][0].reshape(B, C, H * W).permute(0, 2, 1)
        #         x_seg[1][0] = self.attentionhead(x_depth0, x_seg[1][0], x_seg[1][0]).permute(0, 2, 1).reshape(B, C, H, W)
        #         #x_seg[1][0] = self.attentionhead( x_seg[1][0], x_depth0, x_seg[1][0]).permute(0, 2, 1).reshape(B, C, H, W)
        #
        #         x_depth1 = x_depth[1][1]
        #         B, C, H, W = x_depth1.shape
        #         x_depth1 = x_depth1.reshape(B, C, H * W).permute(0, 2, 1)
        #         x_seg[1][1] = x_seg[1][1].reshape(B, C, H * W).permute(0, 2, 1)
        #         x_seg[1][1] = self.attentionhead(x_depth1, x_seg[1][1],  x_seg[1][1]).permute(0, 2, 1).reshape(B, C, H, W)
        #         #x_seg[1][1] = self.attentionhead(x_seg[1][1], x_depth1, x_seg[1][1]).permute(0, 2, 1).reshape(B, C, H, W)
        #
        #         x_depth[1] = tuple(x_depth[1])
        #         x_seg[1] = tuple(x_seg[1])
        #         x_depth = tuple(x_depth)
        #         x_seg = tuple(x_seg)
        #     except:
        #         #pdb.set_trace()
        #         x_depth[0] = list(x_depth[0])
        #         x_seg[0] = list(x_seg[0])
        #
        #         x_depth0 = x_depth[0][0]  # the first dimension: 0 images on kitti 1 images on Cityscape; the second dimension includes images from 4 scales
        #         B, C, H, W = x_depth0.shape
        #         x_depth0 = x_depth0.reshape(B, C, H * W).permute(0, 2, 1)
        #         x_seg[0][0] = x_seg[0][0].reshape(B, C, H * W).permute(0, 2, 1)
        #         x_seg[0][0] = self.attentionhead(x_depth0, x_seg[0][0], x_seg[0][0]).permute(0, 2, 1).reshape(B, C, H, W)
        #         #x_seg[0][0] = self.attentionhead( x_seg[0][0], x_depth0, x_seg[0][0]).permute(0, 2, 1).reshape(B, C, H, W)
        #
        #         x_depth1 = x_depth[0][1]
        #         B, C, H, W = x_depth1.shape
        #         x_depth1 = x_depth1.reshape(B, C, H * W).permute(0, 2, 1)
        #         x_seg[0][1] = x_seg[0][1].reshape(B, C, H * W).permute(0, 2, 1)
        #         x_seg[0][1] = self.attentionhead(x_depth1, x_seg[0][1], x_seg[0][1]).permute(0, 2, 1).reshape(B, C, H, W)
        #         #x_seg[0][1] = self.attentionhead(x_seg[0][1], x_depth1, x_seg[0][1]).permute(0, 2, 1).reshape(B, C, H, W)
        #
        #         x_depth[0] = tuple(x_depth[0])
        #         x_seg[0] = tuple(x_seg[0])
        #         x_depth = tuple(x_depth)
        #         x_seg = tuple(x_seg)

        outputs = list(dict() for _ in batch)

        # All the way back in the loaders each dataset is assigned one or more 'purposes'.
        # For datasets with the 'depth' purpose set the outputs[DATASET_IDX] dict will be
        # populated with depth outputs.
        # Datasets with the 'segmentation' purpose are handled accordingly.
        # If the pose outputs are populated is dependant upon the presence of
        # ('color_aug', -1, 0)/('color_aug', 1, 0) keys in the Dataset.
        for idx, dataset in enumerate(batch):
            if self._check_purposes(dataset, 'depth'):
                x = x_depth[idx]
                x = self.depth(*x)
                if self.semantic_guidance: ## use the output of semantic segmentation to guide the depth estimation
                    semantic_g = x_seg[idx]
                    semantic_g = self.seg(*semantic_g)
                    semantic_x = [0 for x in range (0,4)]
                    new_x = []
                    for res, disp in enumerate(x):
                        semantic_x[res] = disp * semantic_g[res]
                    semantic_x = tuple(semantic_x)
                    semantic_x = self.multires(*semantic_x)
                    for res, disp in enumerate(x):
                        new_x.append(disp + semantic_x[res])
                    #     if res == 0:
                    #         new_x.append(disp + semantic_x[res])
                    #     else:
                    #         new_x.append(0.5*(disp +semantic_x[res] + nn.functional.interpolate(x[res-1],scale_factor=2, mode='bilinear',align_corners = True )))
                    x = tuple(new_x)
                x = reversed(x)  # reverse output list

                for res, disp in enumerate(x):
                    outputs[idx]['disp', res] = disp

            if self._check_purposes(dataset, 'segmentation'):
                x = x_seg[idx]
                if self.depth_guidance: ## use the output of depth estimation to guide the semantic segmentation
                    depth_g = x_depth[idx]
                    depth_g = self.depth(*depth_g)
                    depth_g = depth_g[-1]
                    if self.semantic_guidance:
                        x = self.seg(*x)
                        x = x[-1]
                    else:
                        x= self.seg(*x)
                    x = depth_g * x + x
                elif self.semantic_guidance :
                    x = self.seg(*x)
                    x = x[-1]
                else:
                    x = self.seg(*x)

                outputs[idx]['segmentation_logits', 0] = x

            if self.KEY_FRAME_PREV in dataset:
                frame_prev = dataset[self.KEY_FRAME_PREV]
                frame_cur = dataset[self.KEY_FRAME_CUR]

                # Concatenating joins the previous and the current frame
                # tensors along the first axis/dimension,
                # which is the axis for the color channel
                frame_prev_cur = torch.cat((frame_prev, frame_cur), dim=1)

                outputs[idx]['cam_T_cam', 0, -1] = self.pose(frame_prev_cur, invert=True)

            if self.KEY_FRAME_NEXT in dataset:
                frame_cur = dataset[self.KEY_FRAME_CUR]
                frame_next = dataset[self.KEY_FRAME_NEXT]

                frame_cur_next = torch.cat((frame_cur, frame_next), 1)
                outputs[idx]['cam_T_cam', 0, 1] = self.pose(frame_cur_next, invert=False)

        return tuple(outputs)
