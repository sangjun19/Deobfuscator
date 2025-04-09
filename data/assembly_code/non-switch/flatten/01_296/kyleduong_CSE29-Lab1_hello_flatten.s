	.file	"kyleduong_CSE29-Lab1_hello_flatten.c"
	.text
	.globl	_TIG_IZ_YDPw_argv
	.bss
	.align 8
	.type	_TIG_IZ_YDPw_argv, @object
	.size	_TIG_IZ_YDPw_argv, 8
_TIG_IZ_YDPw_argv:
	.zero	8
	.globl	_TIG_IZ_YDPw_argc
	.align 4
	.type	_TIG_IZ_YDPw_argc, @object
	.size	_TIG_IZ_YDPw_argc, 4
_TIG_IZ_YDPw_argc:
	.zero	4
	.globl	_TIG_IZ_YDPw_envp
	.align 8
	.type	_TIG_IZ_YDPw_envp, @object
	.size	_TIG_IZ_YDPw_envp, 8
_TIG_IZ_YDPw_envp:
	.zero	8
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_YDPw_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_YDPw_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_YDPw_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 108 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-YDPw--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_YDPw_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_YDPw_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_YDPw_envp(%rip)
	nop
	movq	$2, -32(%rbp)
.L11:
	cmpq	$2, -32(%rbp)
	je	.L6
	cmpq	$2, -32(%rbp)
	ja	.L14
	cmpq	$0, -32(%rbp)
	je	.L8
	cmpq	$1, -32(%rbp)
	jne	.L14
	movb	$71, -17(%rbp)
	movb	$111, -16(%rbp)
	movb	$111, -15(%rbp)
	movb	$100, -14(%rbp)
	movb	$98, -13(%rbp)
	movb	$121, -12(%rbp)
	movb	$101, -11(%rbp)
	movb	$33, -10(%rbp)
	movb	$0, -9(%rbp)
	movb	$72, -24(%rbp)
	movb	$101, -23(%rbp)
	movb	$108, -22(%rbp)
	movb	$108, -21(%rbp)
	movb	$111, -20(%rbp)
	movb	$33, -19(%rbp)
	movb	$0, -18(%rbp)
	leaq	-17(%rbp), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -32(%rbp)
	jmp	.L9
.L8:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L12
	jmp	.L13
.L6:
	movq	$1, -32(%rbp)
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
