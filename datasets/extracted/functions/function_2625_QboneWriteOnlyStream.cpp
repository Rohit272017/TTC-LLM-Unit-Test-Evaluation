#include "quiche/quic/qbone/qbone_stream.h"
#include "absl/strings/string_view.h"
#include "quiche/quic/core/quic_data_reader.h"
#include "quiche/quic/core/quic_data_writer.h"
#include "quiche/quic/core/quic_types.h"
#include "quiche/quic/qbone/qbone_constants.h"
#include "quiche/quic/qbone/qbone_session_base.h"
#include "quiche/common/platform/api/quiche_command_line_flags.h"
DEFINE_QUICHE_COMMAND_LINE_FLAG(int, qbone_stream_ttl_secs, 3,
                                "The QBONE Stream TTL in seconds.");
namespace quic {
QboneWriteOnlyStream::QboneWriteOnlyStream(QuicStreamId id,
                                           QuicSession* session)
    : QuicStream(id, session, false, WRITE_UNIDIRECTIONAL) {
  MaybeSetTtl(QuicTime::Delta::FromSeconds(
      quiche::GetQuicheCommandLineFlag(FLAGS_qbone_stream_ttl_secs)));
}
void QboneWriteOnlyStream::WritePacketToQuicStream(absl::string_view packet) {
  WriteOrBufferData(packet,  true, nullptr);
}
QboneReadOnlyStream::QboneReadOnlyStream(QuicStreamId id,
                                         QboneSessionBase* session)
    : QuicStream(id, session,
                 false, READ_UNIDIRECTIONAL),
      session_(session) {
  MaybeSetTtl(QuicTime::Delta::FromSeconds(
      quiche::GetQuicheCommandLineFlag(FLAGS_qbone_stream_ttl_secs)));
}
void QboneReadOnlyStream::OnDataAvailable() {
  sequencer()->Read(&buffer_);
  if (sequencer()->IsClosed()) {
    session_->ProcessPacketFromPeer(buffer_);
    OnFinRead();
    return;
  }
  if (buffer_.size() > QboneConstants::kMaxQbonePacketBytes) {
    if (!rst_sent()) {
      Reset(QUIC_BAD_APPLICATION_PAYLOAD);
    }
    StopReading();
  }
}
}  