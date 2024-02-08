/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE272_Least_Privilege_Principal__wchar_t_w32SHRegCreateUSKey_07.c
Label Definition File: CWE272_Least_Privilege_Principal.label.xml
Template File: point-flaw-07.tmpl.c
*/
/*
 * @description
 * CWE: 272 Least Privilege Principal
 * Sinks: w32SHRegCreateUSKey
 *    GoodSink: Create a registry key using SHRegCreateUSKeyW() and HKEY_CURRENT_USER
 *    BadSink : Create a registry key using SHRegCreateUSKeyW() and HKEY_LOCAL_MACHINE
 * Flow Variant: 07 Control flow: if(static_five==5) and if(static_five!=5)
 *
 * */

#include "std_testcase.h"

#include <windows.h>
#include <shlwapi.h>
#pragma comment( lib, "shlwapi" )

/* The variable below is not declared "const", but is never assigned
   any other value so a tool should be able to identify that reads of
   this will always give its initialized value. */
static int static_five = 5;

#ifndef OMITBAD

void CWE272_Least_Privilege_Principal__wchar_t_w32SHRegCreateUSKey_07_bad()
{
    if(static_five==5)
    {
#ifdef _WIN32
        {
            wchar_t * key = L"TEST\\TestKey";
            HUSKEY hKey = HKEY_LOCAL_MACHINE;

            /* FLAW: Call SHRegCreateUSKeyW() with HKEY_LOCAL_MACHINE violating the least privilege principal */
            if (SHRegCreateUSKeyW(
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
            wchar_t * key = L"TEST\\TestKey";
            HUSKEY hKey = HKEY_CURRENT_USER;
            /* FIX: Call SHRegCreateUSKeyW() with HKEY_CURRENT_USER */
            if (SHRegCreateUSKeyW(
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

/* good1() uses if(static_five!=5) instead of if(static_five==5) */
static void good1()
{
    if(static_five!=5)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
#ifdef _WIN32
        {
            wchar_t * key = L"TEST\\TestKey";
            HUSKEY hKey = HKEY_LOCAL_MACHINE;
            /* FLAW: Call SHRegCreateUSKeyW() with HKEY_LOCAL_MACHINE violating the least privilege principal */
            if (SHRegCreateUSKeyW(
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
            wchar_t * key = L"TEST\\TestKey";
            HUSKEY hKey = HKEY_CURRENT_USER;

            /* FIX: Call SHRegCreateUSKeyW() with HKEY_CURRENT_USER */
            if (SHRegCreateUSKeyW(
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
    if(static_five==5)
    {
#ifdef _WIN32
        {
            wchar_t * key = L"TEST\\TestKey";
            HUSKEY hKey = HKEY_CURRENT_USER;

            /* FIX: Call SHRegCreateUSKeyW() with HKEY_CURRENT_USER */
            if (SHRegCreateUSKeyW(
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
            wchar_t * key = L"TEST\\TestKey";
            HUSKEY hKey = HKEY_LOCAL_MACHINE;
            /* FLAW: Call SHRegCreateUSKeyW() with HKEY_LOCAL_MACHINE violating the least privilege principal */
            if (SHRegCreateUSKeyW(
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

void CWE272_Least_Privilege_Principal__wchar_t_w32SHRegCreateUSKey_07_good()
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
    CWE272_Least_Privilege_Principal__wchar_t_w32SHRegCreateUSKey_07_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE272_Least_Privilege_Principal__wchar_t_w32SHRegCreateUSKey_07_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
