import React from "react"
import "../index.css"
import Sidenav from "../components/sidebar"

export default function Home() {
    return (
        <div style={{ color: `black`,margin:`3rem auto` }}>
            <Sidenav />
            <div class="main">
                <h1>Stephen Fay</h1>
            </div>
        </div>
    );
}

