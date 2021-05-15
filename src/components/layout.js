import React from "react"
import Sidenav from "./sidebar"

export default function Layout({ children }) {
    return (
        <div>
            <Sidenav />
            <div class="main">
                {children}      
            </div>  
        </div>
    )
}