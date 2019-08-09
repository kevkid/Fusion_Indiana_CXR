#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 16:02:00 2018

@author: kevin
Get images from DB
"""
import pandas as pd
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

def get_images_from_db(subset = "full"):
    Base = automap_base()
    # engine, suppose it has two tables 'user' and 'address' set up
    engine = create_engine('mysql+mysqlconnector://root:toor@127.0.0.1/Pubmed_Images?charset=utf8mb4', pool_recycle=3600) # connect to server
    # reflect the tables
    Base.prepare(engine, reflect=True)
    # mapped classes are now created with names by default
    # matching that of the table name.
    Pubmed_Images = Base.classes.images
    #Start Session
    session = Session(engine)
    
    unlabeled = session.query(Pubmed_Images).filter(Pubmed_Images.class_id == -1)
    labeled = session.query(Pubmed_Images).filter(Pubmed_Images.class_id != -1)
    
    unlabeled_image_records = []
    for record in unlabeled:
        unlabeled_image_records.append(int(record.PMC_id))
    
    labled_image_records = []
    for record in labeled:
        labled_image_records.append(int(record.PMC_id))
    
    unlabeled_to_remove = list(set(unlabeled_image_records).intersection(labled_image_records))
    from sqlalchemy import and_
    #get full, only labeled, or only unlabeled
    if subset == "full":
        image_query = session.query(Pubmed_Images).filter(~and_(Pubmed_Images.class_id == -1,
                              Pubmed_Images.PMC_id.in_(unlabeled_to_remove))).all() #remove the unlabeled images for which we have a label
    elif subset == "labeled":
        image_query = session.query(Pubmed_Images).filter(~and_(Pubmed_Images.class_id == -1,
                              Pubmed_Images.PMC_id.in_(unlabeled_to_remove)), Pubmed_Images.class_id > -1).all() #remove the unlabeled images for which we have a label
    elif subset == "unlabeled":
        image_query = session.query(Pubmed_Images).filter(~and_(Pubmed_Images.class_id == -1,
                              Pubmed_Images.PMC_id.in_(unlabeled_to_remove)), Pubmed_Images.class_id == -1).all() #remove the unlabeled images for which we have a label
    else:
        print("MUST SELECT A SUBSET!")
        return -1
    
    images = pd.DataFrame(columns=['PMC_id', 'fid','label', 'caption', 'location', 'class_id'])
    num_images = len(image_query)
    for i, image in enumerate(image_query):
        if i%500==0:
            print("On image {} out of {}".format(i, num_images))
        images.at[i] = {'PMC_id': image.PMC_id, 'fid': image.fid,'label': image.label, 'caption': image.caption, 'location': image.location, 'class_id': image.class_id}    
    return images