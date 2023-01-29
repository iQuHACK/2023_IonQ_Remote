import cirq
import qiskit
import numpy as np
teamname="Qiao"
task="part 1"


def encode_cirq(image):
    circuit=cirq.Circuit()
    if image[0][0]==0:
        circuit.append(cirq.rx(np.pi).on(cirq.LineQubit(0)))
    return circuit

def encode_qiskit(image):
    import numpy as np
    from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
    image1 = np.array(image)
    image_shape_0 = image1.shape[0]
    image_shape_1 = image1.shape[1]
    intensity = QuantumRegister(size=6, name="intensity")
    bits_intensity = ClassicalRegister(size=6, name="bits_intensity")

    num_qubits = int(np.ceil(np.log2(image_shape_0 * image_shape_1)))
    dim_new = int(np.sqrt(2 ** num_qubits))
    image_large = np.zeros((dim_new, dim_new))
    image_large[0:image_shape_0, 0:image_shape_1] = image
    qubits_index = QuantumRegister(size=num_qubits, name="pixel_indexes")
    bits_index = ClassicalRegister(size=num_qubits, name="bits_pixel_indexes")
    qc = QuantumCircuit(intensity, qubits_index, bits_intensity, bits_index)

    qc.h(qubit=qubits_index)
    qc.barrier()

    n = 1

    num_pixel = 2 ** len(qc.qregs[1])

    aux_bin_list = [bin(i)[2:] for i in range(num_pixel)][
                   : dim_new * dim_new
                   ]
    aux_len_bin_list = [len(binary_num) for binary_num in aux_bin_list]
    max_length = max(aux_len_bin_list)
    binary_list = []

    for bnum in aux_bin_list:
        if len(bnum) < max_length:
            new_binary = ""
            for _ in range(max_length - len(bnum)):
                new_binary += "0"
            new_binary += bnum
            binary_list.append(new_binary)
        else:
            binary_list.append(bnum)

    pixel_intensity = []

    pixels_matrix = image_large.tolist()

    for row in pixels_matrix:
        for entry in row:
            intensity = int(np.round(63 * entry))
            pixel_intensity.append(intensity)

    binary_pixel_intensity = [
        bin(p_intensity)[2:] for p_intensity in pixel_intensity
    ]
    for k, bnum in enumerate(binary_list):

        if binary_pixel_intensity[k] != "0":
            for idx, element in enumerate(bnum[::-1]):
                if element == "0":
                    qc.x(qubit=qc.qregs[1][idx])

            for idx, element in enumerate(binary_pixel_intensity[k][::-1]):
                if element == "1":
                    qc.mct(
                        control_qubits=qc.qregs[1],
                        target_qubit=qc.qregs[0][idx],
                    )

            for idx, element in enumerate(bnum[::-1]):
                if element == "0":
                    qc.x(qubit=qc.qregs[1][idx])

    qc.barrier()

    return qc



def decode(histogram):
    keys_list = sorted(list(histogram.keys()))

    mylen = len(str(np.amax(keys_list)))
    intensity_strings = []
    for key in keys_list:
        intensity_strings.append("{0:b}".format(key).zfill(mylen))
    # intensity_strings = [key.split(" ")[1] for key in keys_list]

    pixel_intensity = []

    for string in intensity_strings:
        intensity = 0
        for idx, char in enumerate(string):
            if char == "1":
                intensity += 2 ** (5 - idx)
        intensity = intensity / 63
        pixel_intensity.append(intensity)

    len2 = len(pixel_intensity)
    len1 = int(np.sqrt(len2))
    image_fin = np.array(pixel_intensity).reshape([len1, len1])

    myx, myy = [28, 28]
    image = image_fin[0:myx, 0:myy]
    return image