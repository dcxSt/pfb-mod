import React from "react"
import Sidenav from "../components/sidebar"
import {StaticImage} from 'gatsby-plugin-image';
import "../index.css"

var senryu = require('../content/senryu.json');

export default function Hokku() {
    return (
        <div style={{ color:`black`,margin:`3rem auto` }}>
            <Sidenav />
            <div class="main">
                <h1>Senryu</h1>
                <p>Hokku, as R. H. Blyth wrote, tells the truth but not all of it. It excludes everything not useful or appropriate in its effort to take us beyong our everyday view of things. Senryu happily picks up those discards and does something witty with them. </p>
                <p>Hokku endeavors to show us what we do not know but should; enryu brings out everythign we already know but are reluctant to talk about, and shows us how cheap, tawdry, common, yet sometimes very entertaining it all is.</p>
                <p><b>David Coomler</b></p>
                    {
                        senryu.map(el => <div>
                            <p style={{ marginBottom:`1cm` }} />
                            <haiku style={{ color:"white" }}>{el.date}<br/>{el.season}<br/></haiku>
                            <haiku style={{margin:`75px`}}>{el.l1}<br/></haiku>
                            <haiku style={{margin:`75px`}}>{el.l2}<br/></haiku>
                            <haiku style={{margin:`75px`}}>{el.l3}<br/></haiku>
                            <p style={{ marginBottom:`1cm` }} />
                        </div>)
                    }
            </div>
        </div>
    );
}
