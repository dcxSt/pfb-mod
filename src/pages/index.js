import React from "react"
import "../index.css"
import Sidenav from "../components/sidebar"

export default function Home() {
    return (
        <div style={{ color: `black`,margin:`3rem auto` }}>
            <Sidenav />
            <div class="main">
                <h1>Stephen Fay</h1>
                <pprint>
                    Height ~ 1m85, <br/>
                    Hair: brown, <br/>
                    Eyes: grey,<br/>
                    Weight ~ 70kg, <br/>
                    Language: Python, <br/>
                    Favourite math: stoke's theorem, the isomorphism theorems, <br/> 
                    Favourite ensemble: the canonical ensemble,<br/> 
                    Favourite distrubition: Gibbs, <br/> 
                    Favourite physics: Classical physics, <br/> 
                    Longest breath holding record: 3m+ epsilon seconds, <br/> 
                    Unexpected talent: I play the Cello, <br/> 
                    Text editor of choice: vim, vscode with vim keymappings, <br/> 
                    Other hobbies: Jiu-Jitsu, Judo, hiking, running, biking, basking in the sun.<br/>
                    <br/>Currently I'm making a digital filter for 
                    radio telescopes which should help us locate
                    pulsars. The current filters (polyphase filter 
                    banks) produce sharp spikes of noise at certain 
                    frequencies, I'm trying to find a different 
                    filter that will get rid of these spikes.
                </pprint>
            </div>
        </div>
    );
}

