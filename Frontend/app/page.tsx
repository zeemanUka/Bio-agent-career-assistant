import { ChatInterface } from "@/components/chat-interface";
import { API_BASE_URL } from "@/lib/config";

import styles from "./page.module.css";

export default function Home() {
  return (
    <div className={styles.page}>
      <main className={styles.main}>
        <ChatInterface apiBaseUrl={API_BASE_URL} />
      </main>
    </div>
  );
}
