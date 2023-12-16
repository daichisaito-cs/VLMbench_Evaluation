import numpy as np
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int) -> None:
        super().__init__()
        self.d_k = d_k

    def forward(
        self,
        q: torch.Tensor,  # =Q
        k: torch.Tensor,  # =X
        v: torch.Tensor,  # =X
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        scalar = np.sqrt(self.d_k)
        attention_weight = (
            torch.matmul(q, torch.transpose(k, 1, 2)) / scalar
        )  # 「Q * X^T / (D^0.5)」" を計算

        if mask is not None:  # maskに対する処理
            if mask.dim() != attention_weight.dim():
                raise ValueError(
                    "mask.dim != attention_weight.dim, mask.dim={}, attention_weight.dim={}".format(
                        mask.dim(), attention_weight.dim()
                    )
                )
            attention_weight = attention_weight.data.masked_fill_(
                mask, -torch.finfo(torch.float).max
            )

        attention_weight = nn.functional.softmax(
            attention_weight, dim=2
        )  # Attention weightを計算
        return torch.matmul(attention_weight, v)  # (Attention weight) * X により重み付け.


# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model: int, h: int) -> None:
#         super().__init__()
#         self.d_model = d_model
#         self.h = h
#         self.d_k = d_model // h
#         self.d_v = d_model // h

#         #
#         self.W_k = nn.Parameter(
#             torch.Tensor(h, d_model, self.d_k)  # ヘッド数, 入力次元, 出力次元(=入力次元/ヘッド数)
#         )

#         self.W_q = nn.Parameter(
#             torch.Tensor(h, d_model, self.d_k)  # ヘッド数, 入力次元, 出力次元(=入力次元/ヘッド数)
#         )

#         self.W_v = nn.Parameter(
#             torch.Tensor(h, d_model, self.d_v)  # ヘッド数, 入力次元, 出力次元(=入力次元/ヘッド数)
#         )

#         self.scaled_dot_product_attention = ScaledDotProductAttention(self.d_k)

#         self.linear = nn.Linear(h * self.d_v, d_model)

#     def forward(
#         self,
#         q: torch.Tensor,
#         k: torch.Tensor,
#         v: torch.Tensor,
#         mask_3d: torch.Tensor = None,
#     ) -> torch.Tensor:
#         batch_size, seq_len = q.size(0), q.size(1)

#         """repeat Query,Key,Value by num of heads"""
#         q = q.repeat(self.h, 1, 1, 1)  # head, batch_size, seq_len, d_model
#         k = k.repeat(self.h, 1, 1, 1)  # head, batch_size, seq_len, d_model
#         v = v.repeat(self.h, 1, 1, 1)  # head, batch_size, seq_len, d_model

#         """Linear before scaled dot product attention"""
#         q = torch.einsum(
#             "hijk,hkl->hijl", (q, self.W_q)
#         )  # head, batch_size, d_k, seq_len
#         k = torch.einsum(
#             "hijk,hkl->hijl", (k, self.W_k)
#         )  # head, batch_size, d_k, seq_len
#         v = torch.einsum(
#             "hijk,hkl->hijl", (v, self.W_v)
#         )  # head, batch_size, d_k, seq_len

#         """Split heads"""
#         q = q.view(self.h * batch_size, seq_len, self.d_k)
#         k = k.view(self.h * batch_size, seq_len, self.d_k)
#         v = v.view(self.h * batch_size, seq_len, self.d_v)

#         if mask_3d is not None:
#             mask_3d = mask_3d.repeat(self.h, 1, 1)

#         """Scaled dot product attention"""
#         attention_output = self.scaled_dot_product_attention(
#             q, k, v, mask_3d
#         )  # (head*batch_size, seq_len, d_model)

#         attention_output = torch.chunk(attention_output, self.h, dim=0)
#         attention_output = torch.cat(attention_output, dim=2)

#         """Linear after scaled dot product attention"""
#         output = self.linear(attention_output)
#         return output


class AddPositionalEncoding(nn.Module):
    def __init__(
        self, d_model: int, max_len: int, device: torch.device = torch.device("cpu")
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        positional_encoding_weight: torch.Tensor = self._initialize_weight().to(device,non_blocking=True)
        self.register_buffer("positional_encoding_weight", positional_encoding_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.positional_encoding_weight[:seq_len, :].unsqueeze(0)

    def _get_positional_encoding(self, pos: int, i: int) -> float:
        w = pos / (10000 ** (((2 * i) // 2) / self.d_model))
        if i % 2 == 0:
            return np.sin(w)
        else:
            return np.cos(w)

    def _initialize_weight(self) -> torch.Tensor:
        positional_encoding_weight = [
            [self._get_positional_encoding(pos, i) for i in range(1, self.d_model + 1)]
            for pos in range(1, self.max_len + 1)
        ]
        return torch.tensor(positional_encoding_weight).float()


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(nn.functional.relu(self.linear1(x)))

    class FFN(nn.Module):
        def __init__(self, d_model: int, d_ff: int) -> None:
            super().__init__()
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear2(nn.functional.relu(self.linear1(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        heads_num,
        dropout_rate,
        layer_norm_eps,
        cross_attention=False,
    ) -> None:
        super().__init__()

        self.multi_head_attention = nn.MultiheadAttention(
            d_model, heads_num, batch_first=True
        )
        self.dropout_self_attention = nn.Dropout(dropout_rate)
        self.layer_norm_self_attention = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.ffn = FFN(d_model, d_ff)
        self.dropout_ffn = nn.Dropout(dropout_rate)
        self.layer_norm_ffn = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.cross_attention = cross_attention
        if self.cross_attention:
            self.cross_attn = nn.MultiheadAttention(
                d_model, heads_num, batch_first=True
            )
            self.dropout_cross_attention = nn.Dropout(dropout_rate)
            self.layer_norm_cross_attention = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x, q=None, v=None, mask=None):
        if self.cross_attention and q is not None and v is not None:
            x = self.layer_norm_cross_attention(
                self.__cross_attention_block(x, q, v, mask) + x
            )
        else:
            x = self.layer_norm_self_attention(self.__self_attention_block(x, mask) + x)
        x = self.layer_norm_ffn(self.__feed_forward_block(x) + x)
        return x

    def __cross_attention_block(self, x, q, v, mask):
        """
        cross attention block
        """
        x = self.cross_attn(x, q, v, mask)[0]
        return self.dropout_cross_attention(x)

    def __self_attention_block(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        self attention block
        """
        x = self.multi_head_attention(x, x, x, mask)
        return self.dropout_self_attention(x)

    def __feed_forward_block(self, x: torch.Tensor) -> torch.Tensor:
        """
        feed forward block
        """
        return self.dropout_ffn(self.ffn(x))


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # コンフィグから値を抽出
        max_len = config["max_len"]
        d_model = config["d_model"]
        N = config["N"]
        d_ff = config["d_ff"]
        heads_num = config["heads_num"]
        dropout_rate = config["dropout_rate"]
        device = config.get("device", torch.device("cpu"))
        cross_attention = config.get("cross_attention", False)
        layer_norm_eps = config.get("layer_norm_eps", 1e-5)

        # コンポーネントの初期化
        self.positional_encoding = AddPositionalEncoding(d_model, max_len, device)
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    d_ff,
                    heads_num,
                    dropout_rate,
                    layer_norm_eps,
                    cross_attention=cross_attention,
                )
                for _ in range(N)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x = self.embedding(x)
        x = self.positional_encoding(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        return x

    def forward_x(self, x, q, v, mask=None):
        x = self.positional_encoding(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, q, v, mask)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        heads_num: int,
        dropout_rate: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            d_model, heads_num, batch_first=True
        )
        self.dropout_self_attention = nn.Dropout(dropout_rate)
        self.layer_norm_self_attention = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.src_tgt_attention = nn.MultiheadAttention(
            d_model, heads_num, batch_first=True
        )
        self.dropout_src_tgt_attention = nn.Dropout(dropout_rate)
        self.layer_norm_src_tgt_attention = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.ffn = FFN(d_model, d_ff)
        self.dropout_ffn = nn.Dropout(dropout_rate)
        self.layer_norm_ffn = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(
        self,
        tgt: torch.Tensor,  # Decoder input
        src: torch.Tensor,  # Encoder output
        mask_src_tgt: torch.Tensor,
        mask_self: torch.Tensor,
    ) -> torch.Tensor:
        tgt = self.layer_norm_self_attention(
            tgt + self.__self_attention_block(tgt, mask_self)
        )

        x = self.layer_norm_src_tgt_attention(
            tgt + self.__src_tgt_attention_block(src, tgt, mask_src_tgt)
        )

        x = self.layer_norm_ffn(x + self.__feed_forward_block(x))

        return x

    def __src_tgt_attention_block(
        self, src: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return self.dropout_src_tgt_attention(
            self.src_tgt_attention(tgt, src, src, mask)
        )

    def __self_attention_block(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return self.dropout_self_attention(self.self_attention(x, x, x, mask))

    def __feed_forward_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout_ffn(self.ffn(x))


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        tgt_vocab_size: int,
        max_len: int,
        pad_idx: int,
        d_model: int,
        N: int,
        d_ff: int,
        heads_num: int,
        dropout_rate: float,
        layer_norm_eps: float,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        # self.embedding = Embedding(tgt_vocab_size, d_model, pad_idx)
        self.positional_encoding = AddPositionalEncoding(d_model, max_len, device)
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model, d_ff, heads_num, dropout_rate, layer_norm_eps
                )
                for _ in range(N)
            ]
        )

    def forward(
        self,
        tgt: torch.Tensor,  # Decoder input
        src: torch.Tensor,  # Encoder output
        mask_src_tgt: torch.Tensor,
        mask_self: torch.Tensor,
    ) -> torch.Tensor:
        # tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        for decoder_layer in self.decoder_layers:
            tgt = decoder_layer(
                tgt,
                src,
                mask_src_tgt,
                mask_self,
            )
        return tgt


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        max_len: int,
        d_model: int = 512,
        heads_num: int = 8,
        d_ff: int = 2048,
        N: int = 6,
        dropout_rate: float = 0.1,
        layer_norm_eps: float = 1e-5,
        pad_idx: int = 0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.heads_num = heads_num
        self.d_ff = d_ff
        self.N = N
        self.dropout_rate = dropout_rate
        self.layer_norm_eps = layer_norm_eps
        self.pad_idx = pad_idx
        self.device = device

        self.encoder = TransformerEncoder(
            src_vocab_size,
            max_len,
            pad_idx,
            d_model,
            N,
            d_ff,
            heads_num,
            dropout_rate,
            layer_norm_eps,
            device,
        )

        self.decoder = TransformerDecoder(
            tgt_vocab_size,
            max_len,
            pad_idx,
            d_model,
            N,
            d_ff,
            heads_num,
            dropout_rate,
            layer_norm_eps,
            device,
        )

        self.linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        ----------
        src : torch.Tensor
            単語のid列. [batch_size, max_len]
        tgt : torch.Tensor
            単語のid列. [batch_size, max_len]
        """

        # mask
        pad_mask_src = self._pad_mask(src)

        src = self.encoder(src, pad_mask_src)

        # if tgt is not None:

        # target系列の"0(BOS)~max_len-1"(max_len-1系列)までを入力し、"1~max_len"(max_len-1系列)を予測する
        mask_self_attn = torch.logical_or(
            self._subsequent_mask(tgt), self._pad_mask(tgt)
        )
        dec_output = self.decoder(tgt, src, pad_mask_src, mask_self_attn)

        return self.linear(dec_output)

    def _pad_mask(self, x: torch.Tensor) -> torch.Tensor:
        """単語のid列(ex:[[4,1,9,11,0,0,0...],[4,1,9,11,0,0,0...],[4,1,9,11,0,0,0...]...])からmaskを作成する.
        Parameters:
        ----------
        x : torch.Tensor
            単語のid列. [batch_size, max_len]
        """
        seq_len = x.size(1)
        mask = x.eq(self.pad_idx)  # 0 is <pad> in vocab
        mask = mask.unsqueeze(1)
        mask = mask.repeat(1, seq_len, 1)  # (batch_size, max_len, max_len)
        return mask.to(self.device,non_blocking=True)

    def _subsequent_mask(self, x: torch.Tensor) -> torch.Tensor:
        """DecoderのMasked-Attentionに使用するmaskを作成する.
        Parameters:
        ----------
        x : torch.Tensor
            単語のトークン列. [batch_size, max_len, d_model]
        """
        batch_size = x.size(0)
        max_len = x.size(1)
        return (
            torch.tril(torch.ones(batch_size, max_len, max_len)).eq(0).to(self.device,non_blocking=True)
        )
