	.file	"Beebee1201_CS101_main1_flatten.c"
	.text
	.globl	_TIG_IZ_KvS7_argv
	.bss
	.align 8
	.type	_TIG_IZ_KvS7_argv, @object
	.size	_TIG_IZ_KvS7_argv, 8
_TIG_IZ_KvS7_argv:
	.zero	8
	.globl	_TIG_IZ_KvS7_argc
	.align 4
	.type	_TIG_IZ_KvS7_argc, @object
	.size	_TIG_IZ_KvS7_argc, 4
_TIG_IZ_KvS7_argc:
	.zero	4
	.globl	_TIG_IZ_KvS7_envp
	.align 8
	.type	_TIG_IZ_KvS7_envp, @object
	.size	_TIG_IZ_KvS7_envp, 8
_TIG_IZ_KvS7_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"02"
.LC1:
	.string	"w"
.LC2:
	.string	"win.txt"
.LC3:
	.string	"r"
.LC4:
	.string	"lotto.txt"
.LC5:
	.string	"06"
.LC6:
	.string	"04"
	.text
	.globl	main
	.type	main, @function
main:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$160, %rsp
	movl	%edi, -132(%rbp)
	movq	%rsi, -144(%rbp)
	movq	%rdx, -152(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_KvS7_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_KvS7_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_KvS7_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 135 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-KvS7--0
# 0 "" 2
#NO_APP
	movl	-132(%rbp), %eax
	movl	%eax, _TIG_IZ_KvS7_argc(%rip)
	movq	-144(%rbp), %rax
	movq	%rax, _TIG_IZ_KvS7_argv(%rip)
	movq	-152(%rbp), %rax
	movq	%rax, _TIG_IZ_KvS7_envp(%rip)
	nop
	movq	$16, -72(%rbp)
.L33:
	cmpq	$18, -72(%rbp)
	ja	.L36
	movq	-72(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L36-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L36-.L8
	.long	.L15-.L8
	.long	.L36-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L36-.L8
	.long	.L7-.L8
	.text
.L7:
	movq	-104(%rbp), %rdx
	leaq	-48(%rbp), %rax
	movl	$30, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	leaq	-48(%rbp), %rax
	leaq	.LC0(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strstr@PLT
	movq	%rax, -96(%rbp)
	movq	$15, -72(%rbp)
	jmp	.L23
.L11:
	cmpl	$5, -116(%rbp)
	jg	.L24
	movq	$18, -72(%rbp)
	jmp	.L23
.L24:
	movq	$12, -72(%rbp)
	jmp	.L23
.L10:
	cmpq	$0, -96(%rbp)
	je	.L26
	movq	$5, -72(%rbp)
	jmp	.L23
.L26:
	movq	$0, -72(%rbp)
	jmp	.L23
.L13:
	movq	-104(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$13, -72(%rbp)
	jmp	.L23
.L21:
	cmpq	$0, -88(%rbp)
	je	.L28
	movq	$7, -72(%rbp)
	jmp	.L23
.L28:
	movq	$11, -72(%rbp)
	jmp	.L23
.L19:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -64(%rbp)
	movq	-64(%rbp), %rax
	movq	%rax, -112(%rbp)
	leaq	.LC3(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -56(%rbp)
	movq	-56(%rbp), %rax
	movq	%rax, -104(%rbp)
	movq	-104(%rbp), %rdx
	leaq	-48(%rbp), %rax
	movl	$30, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movl	$0, -116(%rbp)
	movq	$14, -72(%rbp)
	jmp	.L23
.L9:
	movq	$3, -72(%rbp)
	jmp	.L23
.L14:
	leaq	-48(%rbp), %rax
	leaq	.LC5(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strstr@PLT
	movq	%rax, -80(%rbp)
	movq	$2, -72(%rbp)
	jmp	.L23
.L15:
	addl	$1, -116(%rbp)
	movq	$14, -72(%rbp)
	jmp	.L23
.L12:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L34
	jmp	.L35
.L17:
	movq	-112(%rbp), %rdx
	leaq	-48(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fputs@PLT
	movq	$9, -72(%rbp)
	jmp	.L23
.L18:
	movq	-112(%rbp), %rdx
	leaq	-48(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fputs@PLT
	movq	$9, -72(%rbp)
	jmp	.L23
.L22:
	leaq	-48(%rbp), %rax
	leaq	.LC6(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strstr@PLT
	movq	%rax, -88(%rbp)
	movq	$1, -72(%rbp)
	jmp	.L23
.L16:
	movq	-112(%rbp), %rdx
	leaq	-48(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fputs@PLT
	movq	$9, -72(%rbp)
	jmp	.L23
.L20:
	cmpq	$0, -80(%rbp)
	je	.L31
	movq	$6, -72(%rbp)
	jmp	.L23
.L31:
	movq	$9, -72(%rbp)
	jmp	.L23
.L36:
	nop
.L23:
	jmp	.L33
.L35:
	call	__stack_chk_fail@PLT
.L34:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
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
