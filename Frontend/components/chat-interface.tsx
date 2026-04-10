"use client";

import {
  type FormEvent,
  type KeyboardEvent,
  startTransition,
  useEffect,
  useEffectEvent,
  useRef,
  useState,
} from "react";

import styles from "./chat-interface.module.css";
import {
  getBackendHealth,
  streamChatMessage,
  type ApiChatMessage,
} from "@/lib/api";

type ChatInterfaceProps = {
  apiBaseUrl: string;
};

type Message = ApiChatMessage & {
  id: string;
  persist?: boolean;
};

type BackendStatus = "checking" | "online" | "offline";

const SUGGESTED_PROMPTS = [
  "What are your core technical strengths?",
  "Tell me about AI systems you have built.",
  "How do you approach engineering leadership?",
  "Which projects best represent your experience?",
];

const INITIAL_MESSAGES: Message[] = [
  {
    id: "welcome",
    role: "assistant",
    content:
      "Ask about skills, experience, projects, or engineering mindset. I’ll reply through the backend API that now powers this frontend.",
    persist: false,
  },
];

function createMessage(role: ApiChatMessage["role"], content: string): Message {
  return {
    id: crypto.randomUUID(),
    role,
    content,
    persist: true,
  };
}

function getStatusLabel(status: BackendStatus) {
  if (status === "online") {
    return "Backend online";
  }

  if (status === "offline") {
    return "Backend offline";
  }

  return "Checking backend";
}

export function ChatInterface({ apiBaseUrl }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>(INITIAL_MESSAGES);
  const [input, setInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const hasConversationStarted =
    isSubmitting || messages.some((message) => message.role === "user");
  const visibleMessages = hasConversationStarted
    ? messages.filter((message) => message.id !== "welcome")
    : messages;
  const streamingMessage = streamingMessageId
    ? messages.find((message) => message.id === streamingMessageId) ?? null
    : null;

  const scrollToLatestMessage = useEffectEvent((behavior: ScrollBehavior) => {
    const transcript = transcriptRef.current;
    if (!transcript) {
      return;
    }

    transcript.scrollTo({
      top: transcript.scrollHeight,
      behavior,
    });
  });

  useEffect(() => {
    scrollToLatestMessage(messages.length > 1 ? "smooth" : "auto");
  }, [messages]);

  useEffect(() => {
    let cancelled = false;

    async function checkBackend() {
      try {
        await getBackendHealth(apiBaseUrl);
        if (!cancelled) {
          setBackendStatus("online");
        }
      } catch {
        if (!cancelled) {
          setBackendStatus("offline");
        }
      }
    }

    void checkBackend();

    return () => {
      cancelled = true;
    };
  }, [apiBaseUrl]);

  function getHistory() {
    return messages
      .filter((message) => message.persist !== false)
      .map(({ role, content }) => ({ role, content }));
  }

  async function submitMessage(rawMessage: string) {
    const trimmedMessage = rawMessage.trim();
    if (!trimmedMessage || isSubmitting) {
      return;
    }

    const history = getHistory();
    const userMessage = createMessage("user", trimmedMessage);
    const assistantMessageId = crypto.randomUUID();

    startTransition(() => {
      setMessages((currentMessages) => [...currentMessages, userMessage]);
      setInput("");
      setError(null);
    });

    setStreamingMessageId(assistantMessageId);
    setIsSubmitting(true);

    try {
      const reply = await streamChatMessage(
        apiBaseUrl,
        trimmedMessage,
        history,
        (delta) => {
          startTransition(() => {
            setMessages((currentMessages) => {
              const existingMessage = currentMessages.find(
                (message) => message.id === assistantMessageId,
              );

              if (!existingMessage) {
                return [
                  ...currentMessages,
                  {
                    id: assistantMessageId,
                    role: "assistant",
                    content: delta,
                    persist: true,
                  },
                ];
              }

              return currentMessages.map((message) =>
                message.id === assistantMessageId
                  ? { ...message, content: `${message.content}${delta}` }
                  : message,
              );
            });
          });
        },
      );

      startTransition(() => {
        setMessages((currentMessages) =>
          currentMessages.map((message) =>
            message.id === assistantMessageId
              ? { ...message, content: reply }
              : message,
          ),
        );
      });

      setBackendStatus("online");
    } catch (submissionError) {
      setBackendStatus("offline");
      startTransition(() => {
        setMessages((currentMessages) =>
          currentMessages.filter((message) => message.id !== assistantMessageId),
        );
      });
      setError(
        submissionError instanceof Error
          ? submissionError.message
          : "The backend request failed.",
      );
    } finally {
      setStreamingMessageId(null);
      setIsSubmitting(false);
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void submitMessage(input);
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submitMessage(input);
    }
  }

  return (
    <section
      className={`${styles.shell} ${
        hasConversationStarted ? styles.shellStarted : ""
      }`}
    >
      {!hasConversationStarted ? (
        <header className={styles.header}>
          <div className={styles.headerCopy}>
            <p className={styles.kicker}>Live Chat</p>
            <h2>Ask the backend about experience, work, and technical depth.</h2>
          </div>
          <span
            className={`${styles.status} ${
              backendStatus === "online"
                ? styles.statusOnline
                : backendStatus === "offline"
                  ? styles.statusOffline
                  : styles.statusChecking
            }`}
          >
            {getStatusLabel(backendStatus)}
          </span>
        </header>
      ) : null}

      {!hasConversationStarted ? (
        <div className={styles.promptStrip}>
          {SUGGESTED_PROMPTS.map((prompt) => (
            <button
              key={prompt}
              type="button"
              className={styles.promptButton}
              disabled={isSubmitting}
              onClick={() => {
                void submitMessage(prompt);
              }}
            >
              {prompt}
            </button>
          ))}
        </div>
      ) : null}

      <div ref={transcriptRef} className={styles.transcript} aria-live="polite">
        {visibleMessages.map((message) => (
          <div
            key={message.id}
            className={`${styles.messageRow} ${
              message.role === "user" ? styles.userRow : styles.assistantRow
            }`}
          >
            <article
              className={`${styles.message} ${
                message.role === "user"
                  ? styles.userMessage
                  : styles.assistantMessage
              }`}
            >
              <p className={styles.messageLabel}>
                {message.role === "user" ? "You" : "Bio Agent"}
              </p>
              <p className={styles.messageBody}>{message.content}</p>
            </article>
          </div>
        ))}

        {isSubmitting && !streamingMessage?.content.trim() ? (
          <div className={`${styles.messageRow} ${styles.assistantRow}`}>
            <article
              className={`${styles.message} ${styles.assistantMessage} ${styles.typingMessage}`}
            >
              <p className={styles.messageLabel}>Bio Agent</p>
              <div className={styles.typingDots} aria-label="Assistant typing">
                <span />
                <span />
                <span />
              </div>
            </article>
          </div>
        ) : null}
      </div>

      <div className={styles.feedbackRow}>
        {error ? (
          <p className={styles.errorText}>{error}</p>
        ) : (
          <p className={styles.helperText}>
            Press Enter to send. Use Shift+Enter for a new line.
          </p>
        )}
      </div>

      <form className={styles.composer} onSubmit={handleSubmit}>
        <label className={styles.composerLabel} htmlFor="chat-message">
          Message
        </label>
        <textarea
          id="chat-message"
          className={styles.textarea}
          value={input}
          disabled={isSubmitting}
          placeholder="Ask about projects, leadership, strengths, or career history..."
          rows={3}
          onChange={(event) => {
            setInput(event.target.value);
          }}
          onKeyDown={handleComposerKeyDown}
        />

        <div className={styles.composerFooter}>
          <p className={styles.footerMeta}>{apiBaseUrl}/api/chat</p>
          <button
            type="submit"
            className={styles.submitButton}
            disabled={isSubmitting || !input.trim()}
          >
            {isSubmitting ? "Thinking..." : "Send message"}
          </button>
        </div>
      </form>
    </section>
  );
}
