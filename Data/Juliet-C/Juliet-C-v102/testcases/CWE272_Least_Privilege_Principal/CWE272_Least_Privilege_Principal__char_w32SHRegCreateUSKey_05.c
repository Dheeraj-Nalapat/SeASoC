/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE272_Least_Privilege_Principal__char_w32SHRegCreateUSKey_05.c
Label Definition File: CWE272_Least_Privilege_Principal.label.xml
Template File: point-flaw-05.tmpl.c
*/
/*
 * @description
 * CWE: 272 Least Privilege Principal
 * Sinks: w32SHRegCreateUSKey
 *    GoodSink: Create a registry key using SHRegCreateUSKeyA() and HKEY_CURRENT_USER
 *    BadSink : Create a registry key using SHRegCreateUSKeyA() and HKEY_LOCAL_MACHINE
 * Flow Variant: 05 Control flow: if(static_t) and if(static_f)
 *
 * */

#include "std_testcase.h"

#include <windows.h>
#include <shlwapi.h>
#pragma comment( lib, "shlwapi" )

/* The two variables below are not defined as "const", but are never
   assigned any other value, so a tool should be able to identify that
   reads of these will always return their initialized values. */
static int static_t = 1; /* true */
static int static_f = 0; /* false */

#ifndef OMITBAD

void CWE272_Least_Privilege_Principal__char_w32SHRegCreateUSKey_05_bad()
{
    if(static_t)
    {
#ifdef _WIN32
        {
            char * key = "TEST\\TestKey";
            HUSKEY hKey = HKEY_LOCAL_MACHINE;

            /* FLAW: Call SHRegCreateUSKeyA() with HKEY_LOCAL_MACHINE violating the least privilege principal */
            if (SHRegCreateUSKeyA(
                key,
                KEY_WRITE,
                NULL,
                &hKey,
                SHREGSET_HKCU) != ERROR_SUCCESS)
            {
                printLine("Registry key could not be created");
            }
            else {
                printLine("Registry key created successfully");
            }
        }
#endif
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
#ifdef _WIN32
        {
            char * key = "TEST\\TestKey";
            HUSKEY hKey = HKEY_CURRENT_USER;
            /* FIX: Call SHRegCreateUSKeyA() with HKEY_CURRENT_USER */
            if (SHRegCreateUSKeyA(
                        key,
                        KEY_WRITE,
                        NULL,
                        &hKey,
                        SHREGSET_HKCU) != ERROR_SUCCESS)
            {
                printLine("Registry key could not be created");
            }
            else
            {
                printLine("Registry key created successfully");
            }
        }
#endif
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(static_f) instead of if(static_t) */
static void good1()
{
    if(static_f)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
#ifdef _WIN32
        {
            char * key = "TEST\\TestKey";
            HUSKEY hKey = HKEY_LOCAL_MACHINE;
            /* FLAW: Call SHRegCreateUSKeyA() with HKEY_LOCAL_MACHINE violating the least privilege principal */
            if (SHRegCreateUSKeyA(
                        key,
                        KEY_WRITE,
                        NULL,
                        &hKey,
                        SHREGSET_HKCU) != ERROR_SUCCESS)
            {
                printLine("Registry key could not be created");
            }
            else
            {
                printLine("Registry key created successfully");
            }
        }
#endif
    }
    else
    {
#ifdef _WIN32
        {
            char * key = "TEST\\TestKey";
            HUSKEY hKey = HKEY_CURRENT_USER;

            /* FIX: Call SHRegCreateUSKeyA() with HKEY_CURRENT_USER */
            if (SHRegCreateUSKeyA(
                key,
                KEY_WRITE,
                NULL,
                &hKey,
                SHREGSET_HKCU) != ERROR_SUCCESS)
            {
                printLine("Registry key could not be created");
            }
            else {
                printLine("Registry key created successfully");
            }
        }
#endif
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(static_t)
    {
#ifdef _WIN32
        {
            char * key = "TEST\\TestKey";
            HUSKEY hKey = HKEY_CURRENT_USER;

            /* FIX: Call SHRegCreateUSKeyA() with HKEY_CURRENT_USER */
            if (SHRegCreateUSKeyA(
                key,
                KEY_WRITE,
                NULL,
                &hKey,
                SHREGSET_HKCU) != ERROR_SUCCESS)
            {
                printLine("Registry key could not be created");
            }
            else {
                printLine("Registry key created successfully");
            }
        }
#endif
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
#ifdef _WIN32
        {
            char * key = "TEST\\TestKey";
            HUSKEY hKey = HKEY_LOCAL_MACHINE;
            /* FLAW: Call SHRegCreateUSKeyA() with HKEY_LOCAL_MACHINE violating the least privilege principal */
            if (SHRegCreateUSKeyA(
                        key,
                        KEY_WRITE,
                        NULL,
                        &hKey,
                        SHREGSET_HKCU) != ERROR_SUCCESS)
            {
                printLine("Registry key could not be created");
            }
            else
            {
                printLine("Registry key created successfully");
            }
        }
#endif
    }
}

void CWE272_Least_Privilege_Principal__char_w32SHRegCreateUSKey_05_good()
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
    CWE272_Least_Privilege_Principal__char_w32SHRegCreateUSKey_05_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE272_Least_Privilege_Principal__char_w32SHRegCreateUSKey_05_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
