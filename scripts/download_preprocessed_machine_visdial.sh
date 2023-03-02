#!/usr/bin/env bash

# Download Faster R-CNN image features for a subset of the Conceptual Captions 12M (3.6M images)
mkdir -p data/cc12m/
mkdir -p data/cc12m/features

# data to generate synthetic visual dialog data
wget https://www.dropbox.com/s/bn62kgs6famil4b/url_to_cap.json -O data/cc12m/url_to_cap.json
wget https://www.dropbox.com/s/6a3ajhsp3ibrkxg/image_id_to_url.json -O data/cc12m/image_id_to_url.json

# download the CC12M's caption data
wget https://www.dropbox.com/s/n4v6swgiwiebylg/captions.tar.xz -O data/cc12m/

# download the CC12M's dialog data
wget https://www.dropbox.com/s/u5sxtk4nayu1dkp/dialogs.tar.xz -O data/cc12m/

# Download Faster R-CNN image features for a subset of the Conceptual Captions 12M (3.6M images)
# 80G chunk (120k images) x 30 = 2.4T (3.6M images)
wget https://www.dropbox.com/s/ei3jqjomenz6eeb/cc12m_img_feat_0.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/35e3cuv1nuhna5c/cc12m_img_feat_1.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/7s5hjfqbl8e8ymk/cc12m_img_feat_2.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/c1n9o6vfgntsky0/cc12m_img_feat_3.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/y6t8a9qa4yy57zp/cc12m_img_feat_4.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/c7odis6odpqt1nv/cc12m_img_feat_5.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/tl2a7elt3jjz7rr/cc12m_img_feat_6.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/abuj6wy4pz507ve/cc12m_img_feat_7.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/oq8hp6thlx7jqr5/cc12m_img_feat_8.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/1ash33gpe1vrbav/cc12m_img_feat_9.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/nuhz4w0cep6nlsb/cc12m_img_feat_10.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/7vr4ze6hezur95c/cc12m_img_feat_11.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/oampitfu8ncw0bb/cc12m_img_feat_12.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/lp58sfk8ccom7yy/cc12m_img_feat_13.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/s3laqfttgmsemcp/cc12m_img_feat_14.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/47e8jpqg6k7uyy2/cc12m_img_feat_15.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/nt19ej5pt0fdh0i/cc12m_img_feat_16.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/pvdtn74zdo4bxbx/cc12m_img_feat_17.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/partlcu4b1fwnh9/cc12m_img_feat_18.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/mdqx8a197zscjuw/cc12m_img_feat_19.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/0bzn5uaqnlau7zh/cc12m_img_feat_20.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/eptlq7ac1nm8hyi/cc12m_img_feat_21.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/868v0r653f9as9c/cc12m_img_feat_22.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/pjbwqls20lx9b3r/cc12m_img_feat_23.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/n5n1spsa3tr3a8i/cc12m_img_feat_24.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/xie4ltzjmizoo2y/cc12m_img_feat_25.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/to1qnpoq0k0n1or/cc12m_img_feat_26.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/ou39lftwtv7v9mz/cc12m_img_feat_27.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/kxjhwvgutphqpjd/cc12m_img_feat_28.lmdb.tar.xz -O data/cc12m/features
wget https://www.dropbox.com/s/ujs0onuzqy4vbvq/cc12m_img_feat_29.lmdb.tar.xz -O data/cc12m/features
