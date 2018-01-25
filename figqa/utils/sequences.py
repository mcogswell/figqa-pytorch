
NULL = 0

def dynamic_rnn(rnn_model, seq_input, seq_lens, initial_state=None):
    '''
    Inputs:
        rnn_model: Any torch.nn RNN model
        seq_input: Input sequence tensor (padded) of size
                   (batch_size, max_sequence_length, embed_size)
        seq_lens: batch_size length torch.LongTensor or numpy array
        initial_state: Initial hidden/cell state of RNN, expected to be
                       the output of the last time steps of another RNN.
                       Hence, its shape is (batch_size, rnn_hidden_size)

    Output:
        A single tensor of shape (batch_size, rnn_hidden_size) corresponding
        to the outputs of the RNN model at the last time step of each input
        sequence

    Credit: Nirbhay Modhe
    '''
    if initial_state is not None:
        assert seq_input.size(0) == initial_state.size(0), \
                        "batch_size mismatch in initial_states"
        assert initial_state.size(1) == rnn_model.hidden_size, \
                        "initial_state hidden_size mismatch with rnn_model"
        # Transform initial_state from (batch_size, rnn_hidden_size)
        # to (num_layers, batch_size, rnn_hidden_size)
        state = initial_state.unsqueeze(0)
        # TODO: is this right??? (not used atm)
        state = state.repeat(rnn_model.num_layers, 1, 1)
        hx = (state, state)
    else:
        hx = None
    outputs, _ = rnn_model(seq_input, hx)
    rnn_hidden_size = outputs.size(2)

    # Select the (T-1)th elements in dim 1 of outputs using (seq_lens-1)
    gather_indices = (seq_lens-1).contiguous().view(-1, 1, 1)
    gather_indices = gather_indices.repeat(1, 1, rnn_hidden_size)
    # 'Gather' gives output of shape (batchSize, 1, rnnHiddenSize)
    return outputs.gather(1, gather_indices).squeeze(1)
