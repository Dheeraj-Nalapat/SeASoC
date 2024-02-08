/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE256_Plaintext_Storage_of_Password__w32_wchar_t_08.c
Label Definition File: CWE256_Plaintext_Storage_of_Password__w32.label.xml
Template File: sources-sink-08.tmpl.c
*/
/*
 * @description
 * CWE: 256 Plaintext Storage of Password
 * BadSource:  Read the password from a file
 * GoodSource: Read the password from the console
 * Sink:
 *    BadSink : Authenticate the user using LogonUserW()
 * Flow Variant: 08 Control flow: if(static_returns_t()) and if(static_returns_f())
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifdef _WIN32
#include <windows.h>
#pragma comment(lib, "advapi32.lib")
#endif

/* The two function below always return the same value, so a tool
   should be able to identify that calls to the functions will always
   return a fixed value. */
static int static_returns_t()
{
    return 1;
}

static int static_returns_f()
{
    return 0;
}

#ifndef OMITBAD

void CWE256_Plaintext_Storage_of_Password__w32_wchar_t_08_bad()
{
    wchar_t * password;
    wchar_t password_buf[100] = L"";
    password = password_buf;
    if(static_returns_t())
    {
        {
            FILE *password_file;
            password_file = fopen("passwords.txt", "r");
            /* FLAW: Read the password from a file */
            fgetws(password, 100, password_file);
            fclose(password_file);
        }
    }
    else
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
    {
#ifdef _WIN32 /* this is WIN32 specific */
        HANDLE pHandle;
        wchar_t * username = L"User";
        wchar_t * domain = L"Domain";
        /* POTENTIAL FLAW: Attempt to login user with password from the source */
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

/* goodG2B1() - use goodsource and badsink by changing the static_returns_t() to static_returns_f() */
static void goodG2B1()
{
    wchar_t * password;
    wchar_t password_buf[100] = L"";
    password = password_buf;
    if(static_returns_f())
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            FILE *password_file;
            password_file = fopen("passwords.txt", "r");
            /* FLAW: Read the password from a file */
            fgetws(password, 100, password_file);
            fclose(password_file);
        }
    }
    else
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
    {
#ifdef _WIN32 /* this is WIN32 specific */
        HANDLE pHandle;
        wchar_t * username = L"User";
        wchar_t * domain = L"Domain";
        /* POTENTIAL FLAW: Attempt to login user with password from the source */
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

/* goodG2B2() - use goodsource and badsink by reversing the blocks in the if statement */
static void goodG2B2()
{
    wchar_t * password;
    wchar_t password_buf[100] = L"";
    password = password_buf;
    if(static_returns_t())
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
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            FILE *password_file;
            password_file = fopen("passwords.txt", "r");
            /* FLAW: Read the password from a file */
            fgetws(password, 100, password_file);
            fclose(password_file);
        }
    }
    {
#ifdef _WIN32 /* this is WIN32 specific */
        HANDLE pHandle;
        wchar_t * username = L"User";
        wchar_t * domain = L"Domain";
        /* POTENTIAL FLAW: Attempt to login user with password from the source */
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

void CWE256_Plaintext_Storage_of_Password__w32_wchar_t_08_good()
{
    goodG2B1();
    goodG2B2();
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
    CWE256_Plaintext_Storage_of_Password__w32_wchar_t_08_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE256_Plaintext_Storage_of_Password__w32_wchar_t_08_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
