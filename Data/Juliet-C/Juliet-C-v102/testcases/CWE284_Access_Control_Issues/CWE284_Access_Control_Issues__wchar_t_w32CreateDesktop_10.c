/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE284_Access_Control_Issues__wchar_t_w32CreateDesktop_10.c
Label Definition File: CWE284_Access_Control_Issues.label.xml
Template File: point-flaw-10.tmpl.c
*/
/*
 * @description
 * CWE: 284 Access Control Issues
 * Sinks: w32CreateDesktop
 *    GoodSink: Create a desktop using CreateDesktopW() without excessive privileges
 *    BadSink : Create a desktop using CreateDesktopW() with excessive privileges
 * Flow Variant: 10 Control flow: if(global_t) and if(global_f)
 *
 * */

#include "std_testcase.h"

#include <windows.h>
#pragma comment( lib, "user32" )

#ifndef OMITBAD

void CWE284_Access_Control_Issues__wchar_t_w32CreateDesktop_10_bad()
{
    if(global_t)
    {
#ifdef _WIN32
        {
            HDESK hDesk;
            wchar_t * dtName = L"DesktopExample";

            /* FLAW: Call CreateDesktopW() with GENERIC_ALL as the 5th parameter */
            hDesk = CreateDesktopW(
                dtName,
                NULL,
                NULL,
                0,
                GENERIC_ALL,
                NULL);

            if (hDesk == NULL)
            {
                printLine("Desktop could not be created");
            }
            else {
                printLine("Desktop created successfully");
                CloseDesktop(hDesk);
            }
        }
#endif
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
#ifdef _WIN32
        {
            HDESK hDesk;
            wchar_t * dtName = L"DesktopExample";
            /* FIX: Call CreateDesktopW() without GENERIC_READ as the 5th parameter to limit access */
            hDesk = CreateDesktopW(
                        dtName,
                        NULL,
                        NULL,
                        0,
                        GENERIC_READ,
                        NULL);
            if (hDesk == NULL)
            {
                printLine("Desktop could not be created");
            }
            else
            {
                printLine("Desktop created successfully");
                CloseDesktop(hDesk);
            }
        }
#endif
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(global_f) instead of if(global_t) */
static void good1()
{
    if(global_f)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
#ifdef _WIN32
        {
            HDESK hDesk;
            wchar_t * dtName = L"DesktopExample";
            /* FLAW: Call CreateDesktopW() with GENERIC_ALL as the 5th parameter */
            hDesk = CreateDesktopW(
                        dtName,
                        NULL,
                        NULL,
                        0,
                        GENERIC_ALL,
                        NULL);
            if (hDesk == NULL)
            {
                printLine("Desktop could not be created");
            }
            else
            {
                printLine("Desktop created successfully");
                CloseDesktop(hDesk);
            }
        }
#endif
    }
    else
    {
#ifdef _WIN32
        {
            HDESK hDesk;
            wchar_t * dtName = L"DesktopExample";

            /* FIX: Call CreateDesktopW() without GENERIC_READ as the 5th parameter to limit access */
            hDesk = CreateDesktopW(
                dtName,
                NULL,
                NULL,
                0,
                GENERIC_READ,
                NULL);

            if (hDesk == NULL)
            {
                printLine("Desktop could not be created");
            }
            else {
                printLine("Desktop created successfully");
                CloseDesktop(hDesk);
            }
        }
#endif
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(global_t)
    {
#ifdef _WIN32
        {
            HDESK hDesk;
            wchar_t * dtName = L"DesktopExample";

            /* FIX: Call CreateDesktopW() without GENERIC_READ as the 5th parameter to limit access */
            hDesk = CreateDesktopW(
                dtName,
                NULL,
                NULL,
                0,
                GENERIC_READ,
                NULL);

            if (hDesk == NULL)
            {
                printLine("Desktop could not be created");
            }
            else {
                printLine("Desktop created successfully");
                CloseDesktop(hDesk);
            }
        }
#endif
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
#ifdef _WIN32
        {
            HDESK hDesk;
            wchar_t * dtName = L"DesktopExample";
            /* FLAW: Call CreateDesktopW() with GENERIC_ALL as the 5th parameter */
            hDesk = CreateDesktopW(
                        dtName,
                        NULL,
                        NULL,
                        0,
                        GENERIC_ALL,
                        NULL);
            if (hDesk == NULL)
            {
                printLine("Desktop could not be created");
            }
            else
            {
                printLine("Desktop created successfully");
                CloseDesktop(hDesk);
            }
        }
#endif
    }
}

void CWE284_Access_Control_Issues__wchar_t_w32CreateDesktop_10_good()
{
    good1();
    good2();
}

#endif /* OMITGOOD */

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    CWE284_Access_Control_Issues__wchar_t_w32CreateDesktop_10_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE284_Access_Control_Issues__wchar_t_w32CreateDesktop_10_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
