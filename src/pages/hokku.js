import React from "react"
import Sidenav from "../components/sidebar"
import {StaticImage} from 'gatsby-plugin-image';
import "../index.css"

var hokku = require('../content/hokku.json');
console.log(hokku);

export default function Hokku() {
    return (
        <div style={{ color:`black`,margin:`3rem auto` }}>
            <Sidenav />
            <div class="main">
                <h1>Hokku</h1>
                    {/* {console.log(hokku)} */}
                    {
                        hokku.map(el => <div>
                            <p style={{ marginBottom:`1cm` }} />
                            <haiku style={{ color:"white" }}>{el.date}<br/></haiku>
                            <haiku style={{margin:`75px`}}>{el.l1}<br/></haiku>
                            <haiku style={{margin:`100px`}}>{el.l2}<br/></haiku>
                            <haiku style={{margin:`125px`}}>{el.l3}<br/></haiku>
                            <p style={{ marginBottom:`1cm` }} />
                        </div>)
                    }

            </div>
            <div class="main">
                <StaticImage
                    src="../images/fly.png"
                    width={40}
                    alt="fly"
                    className="fly"
                />
            </div>
        </div>
    );
}
