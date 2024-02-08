/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE259_Hard_Coded_Password__w32_wchar_t_17.c
Label Definition File: CWE259_Hard_Coded_Password__w32.label.xml
Template File: sources-sink-17.tmpl.c
*/
/*
 * @description
 * CWE: 259 Use of Hard-coded Password
 * BadSource:  Use a hardcoded password
 * GoodSource: Read the password from the console
 * Sink:
 *    BadSink : Authenticate the user using LogonUserW()
 * Flow Variant: 17 Control flow: for loops
 *
 * */

#include "std_testcase.h"

#include <wchar.h>
#define PASSWORD_SZ 100

#define PASSWORD L"ABCD1234!"

#ifdef _WIN32
#include <windows.h>
#pragma comment(lib, "advapi32.lib")
#endif

#ifndef OMITBAD

void CWE259_Hard_Coded_Password__w32_wchar_t_17_bad()
{
    int h,i;
    wchar_t * password;
    wchar_t password_buf[100] = L"";
    password = password_buf;
    for(h = 0; h < 0; h++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            size_t password_len = 0;
            /* FIX: Read the password from the console */
            fgetws(password, 100, stdin);
            /* Remove the carriage return from the string that is inserted by fgetws() */
            password_len = wcslen(password);
            if (password_len > 0)
            {
                password[password_len-1] = L'\0';
            }
        }
    }
    for(i = 0; i < 1; i++)
    {
        /* FLAW: Use a hardcoded password */
        wcscpy(password, PASSWORD);
    }
    {
#ifdef _WIN32 /* this is WIN32 specific */
        HANDLE pHandle;
        wchar_t * username = L"User";
        wchar_t * domain = L"Domain";
        /* POTENTIAL FLAW: Attempt to login user with password from the source (which may be hardcoded) */
        if (LogonUserW(
                    username,
                    domain,
                    password,
                    LOGON32_LOGON_NETWORK,
                    LOGON32_PROVIDER_DEFAULT,
                    &pHandle) != 0)
        {
            printLine("User logged in successfully.");
            CloseHandle(pHandle);
        }
        else
        {
            printLine("Unable to login.");
        }
#endif
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() - use goodsource and badsink by changing the conditions on the for statements */
static void goodG2B()
{
    int h,i;
    wchar_t * password;
    wchar_t password_buf[100] = L"";
    password = password_buf;
    for(h = 0; h < 1; h++)
    {
        {
            size_t password_len = 0;
            /* FIX: Read the password from the console */
            fgetws(password, 100, stdin);
            /* Remove the carriage return from the string that is inserted by fgetws() */
            password_len = wcslen(password);
            if (password_len > 0)
            {
                password[password_len-1] = L'\0';
            }
        }
    }
    for(i = 0; i < 0; i++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FLAW: Use a hardcoded password */
        wcscpy(password, PASSWORD);
    }
    {
#ifdef _WIN32 /* this is WIN32 specific */
        HANDLE pHandle;
        wchar_t * username = L"User";
        wchar_t * domain = L"Domain";
        /* POTENTIAL FLAW: Attempt to login user with password from the source (which may be hardcoded) */
        if (LogonUserW(
                    username,
                    domain,
                    password,
                    LOGON32_LOGON_NETWORK,
                    LOGON32_PROVIDER_DEFAULT,
                    &pHandle) != 0)
        {
            printLine("User logged in successfully.");
            CloseHandle(pHandle);
        }
        else
        {
            printLine("Unable to login.");
        }
#endif
    }
}

void CWE259_Hard_Coded_Password__w32_wchar_t_17_good()
{
    goodG2B();
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
    CWE259_Hard_Coded_Password__w32_wchar_t_17_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE259_Hard_Coded_Password__w32_wchar_t_17_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
