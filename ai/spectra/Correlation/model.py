import torch
from torch import nn


class SpectraCorrelationNet(nn.Module):
    """
    Rede correlacional da Spectra.

    Essa rede foi pensada para uma fase futura do projeto, em que a Spectra
    não analisa apenas um frame isolado, mas uma sequência de informações
    ao longo do tempo.

    Ela pode receber, por exemplo:
    - labels visuais previstas pela SpectraVisionNet
    - embeddings visuais intermediários
    - informações de fala do Whisper
    - timestamps
    - indicadores de pausa
    - possíveis nomes citados nas falas

    Entrada esperada:
        Tensor no formato [batch_size, sequence_length, input_size]

    Saída:
        Tensor no formato [batch_size, output_size]

    Exemplo de uso futuro:
        sequência de cenas -> rede correlacional -> contexto/personagem/ação relevante
    """

    def __init__(
        self,
        input_size,
        hidden_size=128,
        output_size=64,
        num_layers=1,
        dropout_rate=0.3,
        bidirectional=True,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional

        gru_dropout = dropout_rate if num_layers > 1 else 0.0

        self.sequence_encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=bidirectional,
        )

        direction_multiplier = 2 if bidirectional else 1

        self.context_head = nn.Sequential(
            nn.Linear(hidden_size * direction_multiplier, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, output_size),
        )

    def forward(self, x):
        """
        Processa uma sequência de vetores.

        x:
            [batch_size, sequence_length, input_size]

        retorno:
            [batch_size, output_size]
        """
        sequence_output, hidden_state = self.sequence_encoder(x)

        # Pega a saída do último passo temporal.
        last_step_output = sequence_output[:, -1, :]

        output = self.context_head(last_step_output)

        return output

    def encode_sequence(self, x):
        """
        Retorna a representação contextual da sequência inteira.

        Útil caso, no futuro, você queira usar essa representação para
        outras tarefas, como identificar personagens ou detectar continuidade.
        """
        sequence_output, hidden_state = self.sequence_encoder(x)

        return sequence_output, hidden_state
