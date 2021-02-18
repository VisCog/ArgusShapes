import numpy as np
import pandas as pd
import os.path as osp
import h5py
import argus_shapes as shapes


def df2hdf(Xyy, subjectdata, hdf_file):
    """Converts the data from Pandas DataFrame to HDF5"""
    Xy = Xyy.copy()

    Xy['image'] = pd.Series([resize(row.image, 
                                    (row.img_shape[0] // 2, row.img_shape[1] // 2))
                             for (_, row) in Xy.iterrows()], 
                            index=Xy.index)

    # Convert images into black and white:
    Xy.image = Xy.image.apply(lambda x: x.astype(np.bool))
    
    # Split 'img_shape' into 'img_shape_x' and 'img_shape_y':
    Xy['img_shape_x'] = Xy.img_shape.apply(lambda x: x[0])
    Xy['img_shape_y'] = Xy.img_shape.apply(lambda x: x[1])
    Xy.drop(columns='img_shape', inplace=True)
    
    Xy['xrange_x'] = 0
    Xy['xrange_y'] = 0
    Xy['yrange_x'] = 0
    Xy['yrange_y'] = 0
    for subject_id, row in subjectdata.iterrows():
        idx = Xy.subject == subject_id
        Xy.loc[idx, 'xrange_x'] = row['xrange'][0]
        Xy.loc[idx, 'xrange_y'] = row['xrange'][1]
        Xy.loc[idx, 'yrange_x'] = row['yrange'][0]
        Xy.loc[idx, 'yrange_y'] = row['yrange'][1]
        
    file = h5py.File(hdf_file, 'w')
    for subject, data in Xy.groupby('subject'):
        # Image data:
        file.create_dataset("%s.image" % subject,
                            data=np.array([row.image
                                           for (_, row) in data.iterrows()]))
        # String data:
        for col in ['subject', 'filename', 'stim_class', 'electrode', 'date']:
            dt = h5py.string_dtype(encoding='utf-8')
            file.create_dataset("%s.%s" % (subject, col),
                                data=np.array([row[col]
                                               for (_, row) in data.iterrows()],
                                              dtype=dt))
        # Int data:
        for col in ['area', 'img_shape_x', 'img_shape_y']:
            file.create_dataset("%s.%s" % (subject, col),
                                data=np.array([row[col]
                                               for (_, row) in data.iterrows()],
                                              dtype=np.int64))
        # Float data:
        for col in ['amp', 'freq', 'pdur', 'x_center', 'y_center', 'orientation',
                    'eccentricity', 'compactness', 'xrange_x', 'xrange_y', 
                    'yrange_x', 'yrange_y']:
            file.create_dataset("%s.%s" % (subject, col),
                                data=np.array([row[col]
                                               for (_, row) in data.iterrows()],
                                              dtype=np.float64))
    file.close()


def hdf2df(hdf_file):
    """Converts the data from HDF5 to a Pandas DataFrame"""
    f = h5py.File(hdf_file, 'r')
    
    # Fields names are 'subject.field_name', so we split by '.'
    # to find the subject ID:
    subjects = np.unique([k.split('.')[0] for k in f.keys()])
    
    # Create a DataFrame for every subject, then concatenate:
    dfs = []
    for subject in subjects:
        df = pd.DataFrame()
        df['subject'] = subject
        for key in f.keys():
            if subject not in key:
                continue
            # Find the field name, that's the DataFrame column:
            col = key.split('.')[1]
            if col == 'image':
                # Images need special treatment:
                # - Direct assign confuses Pandas, need a loop
                # - Convert back to float so scikit_image can handle it
                df['image'] = [img.astype(np.float64) for img in f[key]]
            else:
                df[col] = f[key]
        dfs.append(df)
    dfs = pd.concat(dfs)
    f.close()
    
    # Combine 'img_shape_x' and 'img_shape_y' back into 'img_shape' tuple
    dfs['img_shape'] = dfs.apply(lambda x: (x['img_shape_x'], x['img_shape_y']), axis=1)
    dfs['xrange'] = dfs.apply(lambda x: (x['xrange_x'], x['xrange_y']), axis=1)
    dfs['yrange'] = dfs.apply(lambda x: (x['yrange_x'], x['yrange_y']), axis=1)
    dfs.drop(columns=['img_shape_x', 'img_shape_y', 'xrange_x', 'xrange_y', 'yrange_x', 'yrange_y'], inplace=True)
    return dfs.reset_index()


def main():
    # Automatically download data from OSF:
    subjectdata = shapes.load_subjects(osp.join('argus_shapes', 'subjects.csv'))
    Xy = shapes.load_data(osp.join('argus_shapes', 'drawings_single.csv'), random_state=None)

    hdf_file = 'argus_shapes.h5'
    df2hdf(Xy, subjectdata, hdf_file)
    df = hdf2df(hdf_file)

    
if __name__ == "__main__":
    main()
