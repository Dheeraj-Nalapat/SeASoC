/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE620_Unverified_Password_Change__w32_10.c
Label Definition File: CWE620_Unverified_Password_Change__w32.label.xml
Template File: point-flaw-10.tmpl.c
*/
/*
 * @description
 * CWE: 620 Unverified Password Change
 * Sinks: w32NetUserSetInfo
 *    GoodSink: Require old password to set new password using NetUserChangePassword()
 *    BadSink : Set new password without verifying the old one using NetUserSetInfo()
 * Flow Variant: 10 Control flow: if(global_t) and if(global_f)
 *
 * */

#include "std_testcase.h"

#include <windows.h>
#include <lm.h>
#pragma comment(lib, "netapi32.lib")
#define USERNAME L"cwe620testuser"

#ifndef OMITBAD

void CWE620_Unverified_Password_Change__w32_10_bad()
{
    if(global_t)
    {
        {
            wchar_t newpassword[256];
            USER_INFO_1003 my_user_info_1003;
            NET_API_STATUS status;
            my_user_info_1003.usri1003_password = newpassword;
            printWLine(L"Enter new password: ");
            fwscanf(stdin, L"%255s", my_user_info_1003.usri1003_password);
            /* FLAW: Set password without verifying the old one */
            status = NetUserSetInfo(NULL, USERNAME, 1003, (LPBYTE)&my_user_info_1003, NULL);
            if(status == NERR_Success)
            {
                printWLine(L"Success!");
            }
            else
            {
                wprintf(L"NetUserSetInfo failed.  Status = %d = 0x%x\n", status, status);
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            wchar_t oldpassword[256];
            wchar_t newpassword[256];
            NET_API_STATUS status;
            printWLine(L"Enter old password: ");
            fwscanf(stdin, L"%255s", oldpassword);
            printWLine(L"Enter new password: ");
            fwscanf(stdin, L"%255s", newpassword);
            /* FIX: Verify the old password when setting the new password */
            status = NetUserChangePassword(NULL, USERNAME, oldpassword, newpassword);
            if(status == NERR_Success)
            {
                printWLine(L"Success!");
            }
            else
            {
                wprintf(L"NetUserChangePassword failed.  Status = %d = 0x%x\n", status, status);
            }
        }
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
        {
            wchar_t newpassword[256];
            USER_INFO_1003 my_user_info_1003;
            NET_API_STATUS status;
            my_user_info_1003.usri1003_password = newpassword;
            printWLine(L"Enter new password: ");
            fwscanf(stdin, L"%255s", my_user_info_1003.usri1003_password);
            /* FLAW: Set password without verifying the old one */
            status = NetUserSetInfo(NULL, USERNAME, 1003, (LPBYTE)&my_user_info_1003, NULL);
            if(status == NERR_Success)
            {
                printWLine(L"Success!");
            }
            else
            {
                wprintf(L"NetUserSetInfo failed.  Status = %d = 0x%x\n", status, status);
            }
        }
    }
    else
    {
        {
            wchar_t oldpassword[256];
            wchar_t newpassword[256];
            NET_API_STATUS status;
            printWLine(L"Enter old password: ");
            fwscanf(stdin, L"%255s", oldpassword);
            printWLine(L"Enter new password: ");
            fwscanf(stdin, L"%255s", newpassword);
            /* FIX: Verify the old password when setting the new password */
            status = NetUserChangePassword(NULL, USERNAME, oldpassword, newpassword);
            if(status == NERR_Success)
            {
                printWLine(L"Success!");
            }
            else
            {
                wprintf(L"NetUserChangePassword failed.  Status = %d = 0x%x\n", status, status);
            }
        }
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(global_t)
    {
        {
            wchar_t oldpassword[256];
            wchar_t newpassword[256];
            NET_API_STATUS status;
            printWLine(L"Enter old password: ");
            fwscanf(stdin, L"%255s", oldpassword);
            printWLine(L"Enter new password: ");
            fwscanf(stdin, L"%255s", newpassword);
            /* FIX: Verify the old password when setting the new password */
            status = NetUserChangePassword(NULL, USERNAME, oldpassword, newpassword);
            if(status == NERR_Success)
            {
                printWLine(L"Success!");
            }
            else
            {
                wprintf(L"NetUserChangePassword failed.  Status = %d = 0x%x\n", status, status);
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            wchar_t newpassword[256];
            USER_INFO_1003 my_user_info_1003;
            NET_API_STATUS status;
            my_user_info_1003.usri1003_password = newpassword;
            printWLine(L"Enter new password: ");
            fwscanf(stdin, L"%255s", my_user_info_1003.usri1003_password);
            /* FLAW: Set password without verifying the old one */
            status = NetUserSetInfo(NULL, USERNAME, 1003, (LPBYTE)&my_user_info_1003, NULL);
            if(status == NERR_Success)
            {
                printWLine(L"Success!");
            }
            else
            {
                wprintf(L"NetUserSetInfo failed.  Status = %d = 0x%x\n", status, status);
            }
        }
    }
}

void CWE620_Unverified_Password_Change__w32_10_good()
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
    CWE620_Unverified_Password_Change__w32_10_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE620_Unverified_Password_Change__w32_10_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
