import React from "react"
import {Link} from "gatsby"
import "../index.css"

export default function Sidenav() {
    return (
        <div class="sidenav">
            <Link
            to="/"
            style={{ color:"#777",textDecorationLine:"none",fontFamily:"Special Elite" }}
            activeStyle={{ color:"#000",textDecorationLine:"none",fontFamily:"Special Elite" }}
            >
                Home
            </Link>
            <br></br>
            <Link
            to="/about/"
            style={{ color:"#777",textDecorationLine:"none",fontFamily:"Special Elite" }}
            activeStyle={{ color:"#000",textDecorationLine:"none",fontFamily:"Special Elite" }}
            partiallyActive={true}
            >
                About me
            </Link>
            <br></br>
            <Link
            to="/hokku/"
            style={{ color:"#777",textDecorationLine:"none",fontFamily:"Special Elite" }}
            activeStyle={{ color:"#000",textDecorationLine:"none",fontFamily:"Special Elite" }}
            partiallyActive={true}
            >
                Hokku
            </Link>
            <br></br>
            <Link
            to="/writing/"
            style={{ color:"#777",textDecorationLine:"none",fontFamily:"Special Elite" }}
            activeStyle={{ color:"#000",textDecorationLine:"none",fontFamily:"Special Elite" }}
            partiallyActive={true}
            >
                Writing
            </Link>
            <br></br>
            <Link
            to="/quotes/"
            style={{ color:"#777",textDecorationLine:"none",fontFamily:"Special Elite" }}
            activeStyle={{ color:"#000",textDecorationLine:"none",fontFamily:"Special Elite" }}
            partiallyActive={true}
            >
                Quotes
            </Link>
        </div>
    );
}