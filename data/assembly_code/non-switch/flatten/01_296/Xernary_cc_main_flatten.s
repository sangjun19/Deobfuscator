	.file	"Xernary_cc_main_flatten.c"
	.text
	.globl	_TIG_IZ_YY5p_envp
	.bss
	.align 8
	.type	_TIG_IZ_YY5p_envp, @object
	.size	_TIG_IZ_YY5p_envp, 8
_TIG_IZ_YY5p_envp:
	.zero	8
	.globl	_TIG_IZ_YY5p_argc
	.align 4
	.type	_TIG_IZ_YY5p_argc, @object
	.size	_TIG_IZ_YY5p_argc, 4
_TIG_IZ_YY5p_argc:
	.zero	4
	.globl	_TIG_IZ_YY5p_argv
	.align 8
	.type	_TIG_IZ_YY5p_argv, @object
	.size	_TIG_IZ_YY5p_argv, 8
_TIG_IZ_YY5p_argv:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"Hello frienddddddddddddd. How are you?"
.LC1:
	.string	"%s"
	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	addq	$-128, %rsp
	movl	%edi, -100(%rbp)
	movq	%rsi, -112(%rbp)
	movq	%rdx, -120(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_YY5p_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_YY5p_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_YY5p_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 107 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-YY5p--0
# 0 "" 2
#NO_APP
	movl	-100(%rbp), %eax
	movl	%eax, _TIG_IZ_YY5p_argc(%rip)
	movq	-112(%rbp), %rax
	movq	%rax, _TIG_IZ_YY5p_argv(%rip)
	movq	-120(%rbp), %rax
	movq	%rax, _TIG_IZ_YY5p_envp(%rip)
	nop
	movq	$2, -88(%rbp)
.L11:
	cmpq	$2, -88(%rbp)
	je	.L6
	cmpq	$2, -88(%rbp)
	ja	.L14
	cmpq	$0, -88(%rbp)
	je	.L8
	cmpq	$1, -88(%rbp)
	jne	.L14
	movq	stdin(%rip), %rax
	movl	$0, %ecx
	movl	$2, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	setvbuf@PLT
	movq	stdout(%rip), %rax
	movl	$0, %ecx
	movl	$2, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	setvbuf@PLT
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-80(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$0, -88(%rbp)
	jmp	.L9
.L8:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L12
	jmp	.L13
.L6:
	movq	$1, -88(%rbp)
	jmp	.L9
.L14:
	nop
.L9:
	jmp	.L11
.L13:
	call	__stack_chk_fail@PLT
.L12:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
