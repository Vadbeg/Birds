import h5py
import numpy as np

from tqdm import tqdm


if __name__ == '__main__':
    shapes = [(256, 256), (128, 485), (11, 153)]

    with h5py.File('testfile.hdf5', mode='r') as file:
        shape = (20_000, 128, 128)

        print(file['default'])
        print(file['default'][0])

        dataset = file.create_dataset('default',
                                      shape=shape,
                                      maxshape=(None, None, None),
                                      chunks=True,
                                      dtype=np.float32)
        print(f'Type: {dataset}')
        # exit()

        print(dataset.shape)
        # exit()

        for i in tqdm(range(1000)):
            curr_shape = shapes[i % 3]
            reshape_shape = list(curr_shape)

            if curr_shape[0] < dataset.shape[1]:
                reshape_shape[0] = dataset.shape[1]

            if curr_shape[1] < dataset.shape[2]:
                reshape_shape[1] = dataset.shape[2]

            dataset.resize(size=(dataset.shape[0], *reshape_shape))

            insert_value = np.full(shape=reshape_shape, fill_value=np.inf)

            insert_value[:curr_shape[0], :curr_shape[1]] = np.random.random(curr_shape)
            dataset[i] = insert_value

            print(dataset.shape)
